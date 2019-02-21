#!/bin/bash

if [ ! -d "tmp" ]; then mkdir "tmp"; fi

wget https://www.dropbox.com/s/s2j0nzgw0oe7h8o/Universal%20Dependencies%202.3.zip -P 'tmp'

unzip 'tmp/Universal Dependencies 2.3.zip' -d 'tmp'
tar -xf 'tmp/ud-treebanks-v2.3.tgz' -C 'tmp'
tar -xf 'tmp/ud-tools-v2.3.tgz' -C 'tmp'

rm 'tmp/Universal Dependencies 2.3.zip' 'tmp/ud-treebanks-v2.3.tgz' 'tmp/ud-tools-v2.3.tgz' 'tmp/ud-documentation-v2.3.tgz'
