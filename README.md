# DeepSpeedTutorial
DeepSpeed Tutorial

## 1 setup and install

### 1.1 setup python environment for DeepSpeed [Linux]
```shell
# create python environment
conda create -n DeepSpeed python=3.10 openmpi

# activate environment
conda activate DeepSpeed

# install compiler
conda install compilers sysroot_linux-64==2.17 gcc==11.4 ninja py-cpuinfo libaio pydantic ca-certificates certifi openssl

# install build tools
pip install packaging build wheel setuptools 

# install torch latest
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or torch 1.13.1
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 1.2 install deepspeed for Nvidia GPU [Linux]

```shell
git clone https://github.com/microsoft/DeepSpeed.git

cd DeepSpeed

git checkout v0.14.2

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
pip install ./dist/deepspeed-0.14.2+5f631ab-*-linux_x86_64.whl
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

## 3. contact

E-MAIL: louxiao@i32n.com
