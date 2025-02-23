#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env.sh"

oss login

oss mkdir oss://hy-tmp/

cd ~/cache || exit

folder="outputs"

if [ -d "outputs" ]; then
    # file="result-$(date "+%Y%m%d-%H%M%S").zip"
    file="${folder}.zip"
    zip -q -r "${file}" outputs
    # tar -zcf outputs.tar.gz outputs
    oss cp $file oss://hy-tmp/
    if [ $? -eq 0 ]; then
        rm -f $file
    fi
fi

# shutdown