#!/usr/bin/env bash


which deepspeed
which python

unset CUDA_VISIBLE_DEVICES
export PORT=$(( 20000 + $(id -u )))
deepspeed --include localhost:"6,7"  --master_port "$PORT" ds_ex1.py --deepspeed