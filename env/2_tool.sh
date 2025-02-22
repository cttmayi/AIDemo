#!/usr/bin/env bash

if [ ! -d "$LOCAL_TOOL/hfd.sh" ]; then
    mkdir -p $LOCAL_TOOL
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod a+x ./hfd.sh
    mv ./hfd.sh $LOCAL_TOOL
    pip install aria2
fi


