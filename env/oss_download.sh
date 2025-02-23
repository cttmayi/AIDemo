#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env.sh"

oss login

cd ~/cache || exit

# 定义函数
copy() {
    local folder="$1"  # 使用局部变量接收传入的参数
    local file="${folder}.zip"

    # 从 OSS 下载文件
    oss cp "oss://hy-tmp/${file}" .
    if [ -f "$file" ]; then
        unzip -q "$file"  # 解压文件
        rm "$file"        # 删除下载的 zip 文件
    else
        echo "Error: File $file does not exist after download."
        return 1  # 返回错误状态码
    fi
}

copy "outputs"
