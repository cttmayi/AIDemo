#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env.sh"

oss login

cd $LOCAL_DIR || exit

# 定义函数
donwload() {
    local folder="$1"  # 使用局部变量接收传入的参数
    local file="${folder}.tar"
    echo "Downloading $file from OSS..."

    # 从 OSS 下载文件
    oss cp "oss://hy-tmp/${file}" .
    if [ -f "$file" ]; then
        # unzip -q "$file"  # 解压文件
        tar -xf "$file"
        rm "$file"        # 删除下载的 zip 文件
    else
        echo "Error: File $file does not exist after download."
        return 1  # 返回错误状态码
    fi
}


donwload "outputs" &

if [ ! -d "models" ]; then
    donwload "models" &
fi

if [ ! -d "datasets" ]; then
    donwload "datasets" &
fi

wait
echo "All files downloaded successfully."
