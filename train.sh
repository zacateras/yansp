#!/bin/bash

# lang='es'; tb_name='Spanish-GSD'; tb_tag='es_gsd'; word_max_length=60; sent_max_length=155; batch_limit_dev=10000;
lang='pl'; tb_name='Polish-LFG'; tb_tag='pl_lfg'; word_max_length=45; sent_max_length=35; batch_limit_dev=999999999;

python main.py \
  --lang "${lang}" \
  --mode 'train' \
  --signature_prefix "${tb_tag}.ft" \
  --wordvec_file "tmp/ft/cc.${lang}.300.vec.gz" \
  --train_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-train.conllu" \
  --dev_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-dev.conllu" \
  --test_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-test.conllu" \
  --save_dir "tmp/out" \
  --epochs 200 \
  --checkpoint_rolling True \
  --batch_size 2000 \
  --batch_per_epoch 400 \
  --batch_size_dev 2000 \
  --batch_limit_dev "${batch_limit_dev}" \
  --batch_per_console_summary 50 \
  --model_word_dense_size None \
  --model_word_max_length "${word_max_length}" \
  --model_char_embedding_dim 60 \
  --model_char_conv_layers 3 \
  --model_char_conv_size 30 \
  --model_core_type transformer \
  \
  --model_core_transformer_hidden_size 256 \
  --model_core_transformer_use_timing_signal True \
  --model_core_transformer_sent_max_length "${sent_max_length}" \
  --model_core_transformer_layers 3 \
  --model_core_transformer_attention_key_dense_size 128 \
  --model_core_transformer_attention_value_dense_size 128 \
  --model_core_transformer_attention_heads 16 \
  --model_core_transformer_pff_layers 2 \
  --model_core_transformer_pff_filter_size 256 \
  \
  --model_core_bilstm_layers 3 \
  --model_core_bilstm_layer_size 64 \
  --model_core_bilstm_noise 0.2 \
  --model_core_bilstm_layer_dropout 0.2 \
  --model_core_bilstm_dropout 0.2 \
  \
  --model_head_dense_size 256 \
  --model_deprel_dense_size 50 \
  --model_upos_dense_size 50 \
  --model_feats_dense_size 512 \
  --model_lemma_dense_size 50 \
  --model_dropout 0.2 \
  --model_noise 0.2
  #--optimizer_lr 0.01 \
  #--signature_suffix lr.01 \
