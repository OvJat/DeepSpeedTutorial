#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Support Python 3.8
    @author: Lou Xiao(louxiao@i32n.com)
    @maintainer: Lou Xiao(louxiao@i32n.com)
    @copyright: Copyright 2018~2023
    @created time: 2023-09-07 16:51:48 CST
    @updated time: 2023-09-07 16:51:48 CST
"""

import json
import os.path

deepspeed_config = {
    # "train_batch_size": 128,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-8,
            "weight_decay": 3e-7
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 1000
        }
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": False,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": True,
    },
    "fp16": {
        "enabled": True,
        "auto_cast": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "consecutive_hysteresis": False,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "contiguous_gradients": True,
        "overlap_comm": True,
    },
}

config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepspeed_config.json")
with open(config_file, 'w') as f:
    f.write(json.dumps(deepspeed_config, indent=4, ensure_ascii=False))
