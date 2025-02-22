#!/usr/bin/env bash

# use source ./0_env.sh to load the env variables

export LOCAL_TOOL=~/tool

mkdir -p $LOCAL_TOOL

# 如果PATH没有包含 ~/bin，则添加

if [[ ":$PATH:" != *":$LOCAL_TOOL:"* ]]; then
    export PATH=$LOCAL_TOOL:$PATH
    #echo "export PATH=$PATH"
    echo "OK"
fi

# HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 配置临时目录
export CACHE_DIR=/hy-tmp


