#!/bin/bash

if [ ! -d "tmp/ft" ]; then mkdir -p "tmp/ft"; fi

for entry in "$@"
do
    entry=(${entry//,/ })
    lang=${entry[0]}
    type=${entry[1]}
    if [ -z "$type" ]; then type="vec"; fi

    wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.$lang.300.$type.gz -P 'tmp/ft/'
done