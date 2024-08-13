# DeepSpeedTutorial
DeepSpeed Tutorial

NOTICE: This README.md is primarily for (debian-12 and cudatoolkit=12.4). This may be adapted for other cases, but modifications are required.

## 1 setup and install

### 1.1 setup python environment for DeepSpeed [Linux]
```shell
# create python environment
conda create -n DeepSpeed python=3.12 openmpi numpy=1.26

# activate environment
conda activate DeepSpeed

# install compiler
conda install compilers sysroot_linux-64==2.17 gcc==11.4 ninja py-cpuinfo libaio pydantic ca-certificates certifi openssl

# install build tools
pip install packaging build wheel setuptools loguru

# install torch latest
pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
```

### 1.2 install deepspeed for Nvidia GPU [Linux]

```shell
git clone https://github.com/microsoft/DeepSpeed.git

cd DeepSpeed

git checkout v0.14.4

# make deepspeed package
# !!! below is single command
unset CUDA_VISIBLE_DEVICES
TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6;8.9" \
DS_BUILD_CCL_COMM=1 \
DS_BUILD_CPU_ADAM=1 \
DS_BUILD_CPU_ADAGRAD=1 \
DS_BUILD_FUSED_ADAM=1 \
DS_BUILD_FUSED_LAMB=1 \
DS_BUILD_UTILS=1 \
python setup.py build_ext -j24 bdist_wheel

# install deepspeed package
pip install ./dist/deepspeed-0.14.4*linux_x86_64.whl
```

some precompiled wheels:

| OS            |torch version| cuda version | download link                                                                                   |
|:-------------:|:-----------:|:------------:|:-----------------------------------------------------------------------------------------------:|
|CentOS 7 x86_64| 1.13.1      | cuda11.8     | [wheel](wheels/cenots-7-x86_64/Torch1.13.1/deepspeed-0.9.5+fc9e1ee-cp310-cp310-linux_x86_64.whl)|
|CentOS 7 x86_64| 2.0.1       | cuda11.8     | [wheel](wheels/cenots-7-x86_64/Torch2.0.1/deepspeed-0.9.5+fc9e1ee-cp310-cp310-linux_x86_64.whl) |
|Ubuntu 20.04 x86_64| 1.13.1  | cuda11.8     | [wheel](wheels/ubuntu-2004-x86_64/Torch1.13.1/deepspeed-0.9.5+fc9e1ee0-cp310-cp310-linux_x86_64.whl)|
|Ubuntu 20.04 x86_64| 2.0.1   | cuda11.8     | [wheel](wheels/ubuntu-2004-x86_64/Torch2.0.1/deepspeed-0.9.5+fc9e1ee0-cp310-cp310-linux_x86_64.whl)|


## 2. running

```shell
git clone https://github.com/OvJat/DeepSpeedTutorial.git
cd DeepSpeedTutorial
# editor hostfile 
./run_dist.sh &
disown 
# check logging
tail -f ./logs/run_dist.sh.log
```
## 2.1 debug and faq
The new versions of CUDA and PyTorch require the following environment to be set up:
- NCCL_SOCKET_IFNAME. to find the proper value, using command line: `ip addr`
- NCCL_IB_DISABLE. if the network-interface is not InfiniBand, set NCCL_IB_DISABLE=1

## 3. contact

E-MAIL: louxiao@i32n.com
