#!/usr/bin/env bash

oss login

oss mkdir oss://backup/

if [ -d "/hy-tmp/models" ]; then
    tar -zcf /hy-tmp/models.tar.gz /hy-tmp/models
    oss cp /hy-tmp/models.tar.gz oss://backup/
    if [ $? -eq 0 ]; then
        # rm -rf /hy-tmp/models
        rm /hy-tmp/models.tar.gz
    fi
fi

if [ -d "/hy-tmp/datasets" ]; then
    tar -zcf /hy-tmp/datasets.tar.gz /hy-tmp/datasets
    oss cp /hy-tmp/datasets.tar.gz oss://backup/
    if [ $? -eq 0 ]; then
        # rm -rf /hy-tmp/datasets
        rm /hy-tmp/datasets.tar.gz
    fi
fi

if [ -d "/hy-tmp/outputs" ]; then
    tar -zcf /hy-tmp/outputs.tar.gz /hy-tmp/outputs
    oss cp /hy-tmp/outputs.tar.gz oss://backup/
    if [ $? -eq 0 ]; then
        # rm -rf /hy-tmp/outputs
        rm /hy-tmp/outputs.tar.gz
    fi
fi