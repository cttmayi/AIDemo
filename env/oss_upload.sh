#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env.sh"

oss login

oss mkdir oss://hy-tmp/

cd ~/cache || exit

# 定义函数
upload() {
    local folder="$1"  # 使用局部变量接收传入的参数

    # 检查文件夹是否存在
    if [ -d "$folder" ]; then
        local file="${folder}.zip"  # 定义压缩文件名

        # 压缩文件夹
        zip -q -r "${file}" "$folder"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to zip the folder $folder."
            return 1
        fi

        # 上传到 OSS
        oss cp "${file}" "oss://hy-tmp/${file}"
        if [ $? -eq 0 ]; then
            echo "Upload successful: ${file}"
            rm -f "${file}"  # 删除本地压缩文件
        else
            echo "Error: Failed to upload ${file} to OSS."
            return 1
        fi
    else
        echo "Error: Folder $folder does not exist."
        return 1
    fi
}



# 如果第一个参数是 "allls"，则上传所有文件夹 否则只上传指定文件夹
if [ "$1" = "all" ]; then
    for folder in */; do
        folder="${folder%/}"  # 去掉末尾的斜杠
        upload "${folder}"
    done
else
    upload "outputs"
fi



# shutdown