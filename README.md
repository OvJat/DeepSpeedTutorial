# DeepSpeedTutorial
DeepSpeed Tutorial

## 1 setup 

### 1.1 setup python environment [Linux]
```shell
conda create -n Torch python=3.10
conda activate Torch
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

### 1.2 install deepspeed [Linux]

#### setup gcc compiler
```shell
# on CentOS 7
sudo yum install -y centos-release-scl-rh
sudo yum update
sudo yum install -y devtoolset-9
source /opt/rh/devtoolset-9/enable

# on debian 
sudo apt update
sudo apt install build-essential manpages-dev
```


```shell
git clone https://github.com/microsoft/DeepSpeed.git

cd DeepSpeed

git checkout v0.9.5
DS_BUILD_CCL_COMM=1 \
DS_BUILD_CPU_ADAM=1 \
DS_BUILD_CPU_ADAGRAD=1 \
DS_BUILD_FUSED_ADAM=1 \
DS_BUILD_FUSED_LAMB=1 \
DS_BUILD_UTILS=1 \
python setup.py build_ext -j24 bdist_wheel

pip install ./dist/deepspeed-0.9.5+fc9e1ee-*-linux_x86_64.whl
```

some precompiled wheels:

| OS    |torch version| cuda version | download link|
|:---:|:---:|:---:|:---:|
|CentOS 7 x86_64| 1.13.1 | cuda11.8 | [wheel](wheels/cenots-7-x86_64/Torch1.13.1/deepspeed-0.9.5+fc9e1ee-cp310-cp310-linux_x86_64.whl)|
|CentOS 7 x86_64| 2.0.1 | cuda11.8 | [wheel](wheels/cenots-7-x86_64/Torch2.0.1/deepspeed-0.9.5+fc9e1ee-cp310-cp310-linux_x86_64.whl)|


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

## 3. discussion

E-MAIL: louxiao@i32n.com
