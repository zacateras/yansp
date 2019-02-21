#!/bin/bash

if [ ! -d "tmp/ft" ]; then mkdir -p "tmp/ft"; fi

wget https://github.com/facebookresearch/fastText/archive/v0.2.0.zip -P 'tmp/ft/'
unzip 'tmp/ft/v0.2.0.zip' -d 'tmp/ft/'
cd 'tmp/ft/fastText-0.2.0'
make
