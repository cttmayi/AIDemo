#!/usr/bin/env bash

# Install Python 3.11
conda create --name env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate env


pip install --upgrade pip
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

# pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install "unsloth[colab-new] @ git+https://gitclone.com/github.com/unslothai/unsloth.git" # 镜像下载
pip install "unsloth[colab-new] @ git+https://kkgithub.com/unslothai/unsloth.git" # 镜像下载

pip install --no-deps trl peft accelerate bitsandbytes vllm

pip install vllm

pip install --upgrade pillow

pip instlal nvitop

