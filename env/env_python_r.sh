#!/bin/bash

# Install Python 3.11


# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url http://mirrors.aliyun.com/pypi/simple/
pip config set global.trusted-host mirrors.aliyun.com

pip install --upgrade pip

pip install -r requirements.txt


