#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Lou Xiao(louxiao@i32n.com)
# @maintainer: Lou Xiao(louxiao@i32n.com)
# @copyright: Copyright 2018~2024
# @created time: 2024-05-28 15:11:20 CST
# @updated time: 2024-05-28 15:11:20 CST

from typing import Iterator

import torch.utils.data as tud
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from deepspeed import DeepSpeedEngine


class EnhancedDistributedSampler(Sampler):

    def __init__(
            self,
            dataset: tud.Dataset,
            num_replicas: int = None,
            rank: int = None,
            shuffle: bool = True,
            seed: int = 0,
            drop_last: bool = True,
            num_epochs: int = 1,
            skip_step: int = 0,
    ):
        super().__init__(None)
        self._data_sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self._num_epochs = num_epochs
        self._skip_step = skip_step

    @staticmethod
    def _iter_fn(num_epochs: int, data_sampler: DistributedSampler):
        for epoch in range(num_epochs):
            data_sampler.epoch = epoch
            for index in data_sampler:
                yield index

    def __iter__(self) -> Iterator[int]:
        it = iter(self._iter_fn(self._num_epochs, self._data_sampler))
        for _ in range(self._skip_step):
            next(it)
        return it

    def __len__(self):
        return self._num_epochs * self._data_sampler.num_samples


def init_dataloader(engine: DeepSpeedEngine, dataset: tud.Dataset, num_epochs: int = 100):
    data_sampler = EnhancedDistributedSampler(
        dataset=dataset,
        num_replicas=engine.dp_world_size,
        rank=engine.global_rank,
        shuffle=True,
        drop_last=True,
        num_epochs=num_epochs,
        skip_step=(engine.global_samples // engine.dp_world_size),
    )
    engine.training_dataloader = engine.deepspeed_io(dataset, data_sampler=data_sampler, num_local_io_workers=4)
    return engine.training_dataloader
