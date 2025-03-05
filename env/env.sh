#!/bin/bash

# 获取当前脚本的绝对路径
current_script=$(realpath "$0")

# 获取当前脚本所在的目录
current_dir=$(dirname "$current_script")

# 获取上级目录
parent_dir=$(dirname "$current_dir")



export LOCAL_TOOL=~/tool

mkdir -p $LOCAL_TOOL

# 如果PATH没有包含 ~/bin，则添加

if [[ ":$PATH:" != *":$LOCAL_TOOL:"* ]]; then
    export PATH=$LOCAL_TOOL:$PATH
    #echo "export PATH=$PATH"
fi

# HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 配置临时目录
# export LOCAL_DIR=~/cache
export LOCAL_DIR=/hy-tmp


export PYTHONPATH=$parent_dir:$PYTHONPATH

