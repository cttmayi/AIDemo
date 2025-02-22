#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "$SCRIPT_DIR/env.sh"

if [ -z "$LOCAL_TOOL" ]; then
    echo "LOCAL_TOOL is not defined"
    exit 1
fi


if [ ! -d "$LOCAL_TOOL/hfd.sh" ]; then
    mkdir -p $LOCAL_TOOL
    wget https://hf-mirror.com/hfd/hfd.sh -O $LOCAL_TOOL/hfd.sh
    chmod a+x $LOCAL_TOOL/hfd.sh
fi



