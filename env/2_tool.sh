#!/usr/bin/env bash

if [ -z "$LOCAL_TOOL" ]; then
    echo "LOCAL_TOOL is not defined"
    exit 1
fi


if [ ! -d "$LOCAL_TOOL/hfd.sh" ]; then
    mkdir -p $LOCAL_TOOL
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod a+x ./hfd.sh
    mv ./hfd.sh $LOCAL_TOOL
fi

pip install aria2


