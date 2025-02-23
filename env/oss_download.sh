#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "$SCRIPT_DIR/env.sh"

oss login

cd ~/cache || exit

folder="outputs"

file="${folder}.zip"
oss cp "oss://hy-tmp/${file}" .
if [ -f "$file" ]; then
    unzip -q "$file"
    rm "$file"
fi
