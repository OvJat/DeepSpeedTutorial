#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Support Python 3.8
    @author: Lou Xiao(louxiao@i32n.com)
    @maintainer: Lou Xiao(louxiao@i32n.com)
    @copyright: Copyright 2018~2023
    @created time: 2023-09-18 14:07:23 CST
    @updated time: 2023-09-18 14:07:23 CST
"""

import os.path
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torchvision import transforms
import deepspeed

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from loguru import logger as logging

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


def init_dataset(
        dataset_dir: str,
        tokenizer=None,
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = False,
):
    from datasets import load_dataset
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(dataset_dir)
    column_names = dataset["train"].column_names
    image_column, text_column = column_names
    print(column_names)

    # preprocess
    def text_processor(text_list):
        texts = []
        for text in text_list:
            if isinstance(text, str):
                texts.append(text)
            elif isinstance(text, (list, np.ndarray)):
                # take a random caption if there are multiple
                texts.append(random.choice(text))
            else:
                raise ValueError(
                    f"Caption column `{text}` should contain either strings or lists of strings."
                )
        r = tokenizer(
            texts,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return r.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def custom_transform(samples):
        images = [image.convert("RGB") for image in samples[image_column]]
        samples["image"] = [train_transforms(image) for image in images]
        samples["text"] = text_processor(samples[text_column])
        return samples

    dataset = dataset['train'].with_transform(custom_transform)
    return dataset


def collate_fn(samples):
    image = torch.stack([sample["image"] for sample in samples])
    image = image.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["text"] for example in samples])
    return {"image": image, "text": input_ids}


class Diffusion(nn.Module):

    def __init__(self, model_path: str):
        super().__init__()

        # init components
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            local_files_only=True,
            use_auth_token=False,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.vae.decoder = None
        # we're only training this
        self.unet = pipe.unet
        self.noise_scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.tokenizer = pipe.tokenizer

    def forward(self, images, texts):
        # train step
        with torch.no_grad():
            # Convert images to latent space
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # process text
            text_tokens = self.text_encoder(texts)[0]

            noise = torch.randn_like(latents)
            # Sample a random timestep for each image
            batch_size = latents.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.add_noise(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        # Predict the noise residual and compute loss
        predications = self.unet(noisy_latents, timesteps, text_tokens).sample

        loss = tnf.mse_loss(predications, target, reduction="mean")
        return loss


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='deepspeed training script.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # init distributed
    deepspeed.init_distributed()

    # init model
    logging.debug("init model")
    model_dir = os.path.join(SCRIPT_PATH, 'data', 'stable-diffusion-2-1')
    model = Diffusion(model_dir)

    # init dataset
    logging.debug("init dataset")
    dataset_dir = os.path.join(SCRIPT_PATH, 'data', 'pokemon-blip-captions')
    dataset = init_dataset(
        dataset_dir,
        tokenizer=model.tokenizer,
        resolution=512,
        center_crop=False,
        random_flip=False,
    )

    # init engine
    logging.debug("init engine")
    engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.unet.parameters(),
        training_data=dataset,
        collate_fn=collate_fn,
        # config=deepspeed_config,
    )

    # load checkpoint
    engine.load_checkpoint("./data/SavedModels/FineTunedStableDiffusion/")

    # train
    last_time = time.time()
    loss_list = []
    echo_interval = 1

    engine.train()
    for step, samples in enumerate(training_dataloader):
        step += 1
        images = samples['image'].to(device=engine.device, dtype=torch.float16)
        texts = samples['text'].to(device=engine.device, dtype=torch.long)

        loss = engine(images, texts)
        engine.backward(loss)
        engine.step()
        loss_list.append(loss.detach().cpu().numpy())

        if step % echo_interval == 0:
            loss_avg = np.mean(loss_list[-echo_interval:])
            used_time = time.time() - last_time
            time_p_step = used_time / echo_interval
            if args.local_rank == 0:
                logging.info(
                    "[Train Step] Step:{:10d}  Loss:{:8.4f} | Time/Batch: {:6.4f}s",
                    step, loss_avg, time_p_step,
                )
            last_time = time.time()
    # save checkpoint
    engine.save_checkpoint("./data/SavedModels/FineTunedStableDiffusion/")


if __name__ == '__main__':
    main()
