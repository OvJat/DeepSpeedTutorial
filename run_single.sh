#!/usr/bin/env bash


which deepspeed
which python

unset CUDA_VISIBLE_DEVICES
#export PORT=$(( 20000 + $(id -u )))
export PORT=$(python -Bu get_tcp_port.py 2>/dev/null | grep 'Distributed TCP PORT' | awk -F'|' '{print $2}')
echo deepspeed --include localhost:"6,7"  --master_port "$PORT" ds_ex1.py --deepspeed