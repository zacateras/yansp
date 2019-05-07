#!/bin/bash

python main.py \
  evaluate \
  --model_dir $1 \
  --conllu_file $2 \
  --summary_file $3
