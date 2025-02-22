#!/usr/bin/env bash

oss login

oss mkdir oss://hy-tmp/

cd /hy-tmp || exit

if [ -d "models" ]; then
    tar -zcf models.tar.gz models
    oss cp models.tar.gz oss://hy-tmp/
    if [ $? -eq 0 ]; then
        # rm -rf models
        rm models.tar.gz
    fi
fi

if [ -d "datasets" ]; then
    tar -zcf datasets.tar.gz datasets
    oss cp datasets.tar.gz oss://hy-tmp/
    if [ $? -eq 0 ]; then
        # rm -rf datasets
        rm datasets.tar.gz
    fi
fi

if [ -d "outputs" ]; then
    tar -zcf outputs.tar.gz outputs
    oss cp outputs.tar.gz oss://hy-tmp/
    if [ $? -eq 0 ]; then
        # rm -rf outputs
        rm outputs.tar.gz
    fi
fi