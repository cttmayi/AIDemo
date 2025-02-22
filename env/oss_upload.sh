#!/bin/bash

oss login

oss mkdir oss://hy-tmp/

cd /hy-tmp || exit


if [ -d "outputs" ]; then
    file="result-$(date "+%Y%m%d-%H%M%S").zip"
    zip -q -r "${file}" outputs
    # tar -zcf outputs.tar.gz outputs
    oss cp $file oss://hy-tmp/
    if [ $? -eq 0 ]; then
        rm -f $file
    fi
fi

# shutdown