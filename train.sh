#!/bin/bash


lang='ar'; tb_name='Arabic-PADT'; tb_tag='ar_padt'; word_max_length=22; sent_max_length=400; batch_limit_dev=999999999; dropout=0.7; 
lang='cs'; tb_name='Czech-PDT'; tb_tag='cs_pdt'; word_max_length=40; sent_max_length=200; batch_limit_dev=999999999; dropout=0.7; 
lang='de'; tb_name='German-GSD'; tb_tag='de_gsd'; word_max_length=37; sent_max_length=120; batch_limit_dev=999999999; dropout=0.7; 
lang='en'; tb_name='English-EWT'; tb_tag='en_ewt'; word_max_length=35; sent_max_length=170; batch_limit_dev=999999999; dropout=0.7; 
lang='pl'; tb_name='Polish-LFG'; tb_tag='pl_lfg'; word_max_length=30; sent_max_length=100; batch_limit_dev=999999999; dropout=0.7; 
lang='zh'; tb_name='Chinese-GSD'; tb_tag='zh_gsd'; word_max_length=20; sent_max_length=120; batch_limit_dev=999999999; dropout=0.7; 
lang='ru'; tb_name='Russian-SynTagRus'; tb_tag='ru_syntagrus'; word_max_length=50; sent_max_length=210; batch_limit_dev=999999999; dropout=0.7; 
lang='ko'; tb_name='Korean-Kaist'; tb_tag='ko_kaist'; word_max_length=25; sent_max_length=85; batch_limit_dev=999999999; dropout=0.7; 
lang='no'; tb_name='Norwegian-Bokmaal'; tb_tag='no_bokmaal'; word_max_length=75; sent_max_length=125; batch_limit_dev=999999999; dropout=0.7; 
lang='no'; tb_name='Norwegian-Nynorsk'; tb_tag='no_nynorsk'; word_max_length=75; sent_max_length=125; batch_limit_dev=999999999; dropout=0.7; 
lang='es'; tb_name='Spanish-AnCora'; tb_tag='es_ancora'; word_max_length=35; sent_max_length=155; batch_limit_dev=10000; dropout=0.7; 

lang='ru'; tb_name='Russian-SynTagRus'; tb_tag='ru_syntagrus'; word_max_length=50; sent_max_length=210; batch_limit_dev=999999999; dropout=0.7; 
python main.py \
  train \
  --signature_prefix "${tb_tag}.${word_max_length}.${sent_max_length}.ft.d${dropout}" \
  --wordvec_file "tmp/ft/cc.${lang}.300.vec.gz" \
  --train_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-train.conllu" \
  --dev_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-dev.conllu" \
  --test_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-test.conllu" \
  --save_dir "tmp/out" \
  --epochs 200 \
  --epochs_early_stopping 10 \
  --checkpoint_rolling True \
  --batch_size 1500 \
  --batch_per_epoch 400 \
  --batch_per_console_summary 50 \
  --batch_size_dev 1500 \
  --batch_limit_dev "${batch_limit_dev}" \
  --model_inputs char word \
  --model_word_dense_size None \
  --model_word_max_length "${word_max_length}" \
  --model_char_embedding_dim 150 \
  --model_char_conv_layers 3 \
  --model_char_conv_size 150 \
  --model_core_type transformer \
  \
  --model_core_transformer_input_dropout ${dropout} \
  --model_core_transformer_hidden_size 256 \
  --model_core_transformer_use_timing_signal True \
  --model_core_transformer_sent_max_length "${sent_max_length}" \
  --model_core_transformer_layers 3 \
  --model_core_transformer_attention_key_dense_size 128 \
  --model_core_transformer_attention_value_dense_size 128 \
  --model_core_transformer_attention_heads 16 \
  --model_core_transformer_attention_dropout ${dropout} \
  --model_core_transformer_pff_layers 2 \
  --model_core_transformer_pff_filter_size 1024 \
  --model_core_transformer_pff_dropout ${dropout} \
  --model_core_transformer_layer_dropout ${dropout} \
  \
  --model_core_bilstm_layers 2 \
  --model_core_bilstm_layer_size 512 \
  --model_core_bilstm_noise 0.0 \
  --model_core_bilstm_layer_dropout ${dropout} \
  --model_core_bilstm_dropout ${dropout} \
  \
  --model_head_dense_size 256 \
  --model_deprel_dense_size 50 \
  --model_upos_dense_size 50 \
  --model_feats_dense_size 512 \
  --model_lemma_dense_size 50 \
  --model_dropout ${dropout} \
  --model_noise 0.0
  #--optimizer_lr 0.01 \
  #--signature_suffix lr.01 \


lang='zh'; tb_name='Chinese-GSD'; tb_tag='zh_gsd'; word_max_length=20; sent_max_length=120; batch_limit_dev=999999999; dropout=0.7; 
python main.py \
  train \
  --signature_prefix "${tb_tag}.${word_max_length}.${sent_max_length}.ft.d${dropout}" \
  --wordvec_file "tmp/ft/cc.${lang}.300.vec.gz" \
  --train_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-train.conllu" \
  --dev_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-dev.conllu" \
  --test_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-test.conllu" \
  --save_dir "tmp/out" \
  --epochs 200 \
  --epochs_early_stopping 10 \
  --checkpoint_rolling True \
  --batch_size 1500 \
  --batch_per_epoch 400 \
  --batch_per_console_summary 50 \
  --batch_size_dev 1500 \
  --batch_limit_dev "${batch_limit_dev}" \
  --model_inputs char word \
  --model_word_dense_size None \
  --model_word_max_length "${word_max_length}" \
  --model_char_embedding_dim 150 \
  --model_char_conv_layers 3 \
  --model_char_conv_size 150 \
  --model_core_type transformer \
  \
  --model_core_transformer_input_dropout ${dropout} \
  --model_core_transformer_hidden_size 256 \
  --model_core_transformer_use_timing_signal True \
  --model_core_transformer_sent_max_length "${sent_max_length}" \
  --model_core_transformer_layers 3 \
  --model_core_transformer_attention_key_dense_size 128 \
  --model_core_transformer_attention_value_dense_size 128 \
  --model_core_transformer_attention_heads 16 \
  --model_core_transformer_attention_dropout ${dropout} \
  --model_core_transformer_pff_layers 2 \
  --model_core_transformer_pff_filter_size 1024 \
  --model_core_transformer_pff_dropout ${dropout} \
  --model_core_transformer_layer_dropout ${dropout} \
  \
  --model_core_bilstm_layers 2 \
  --model_core_bilstm_layer_size 512 \
  --model_core_bilstm_noise 0.0 \
  --model_core_bilstm_layer_dropout ${dropout} \
  --model_core_bilstm_dropout ${dropout} \
  \
  --model_head_dense_size 256 \
  --model_deprel_dense_size 50 \
  --model_upos_dense_size 50 \
  --model_feats_dense_size 512 \
  --model_lemma_dense_size 50 \
  --model_dropout ${dropout} \
  --model_noise 0.0
  #--optimizer_lr 0.01 \
  #--signature_suffix lr.01 \


lang='ko'; tb_name='Korean-Kaist'; tb_tag='ko_kaist'; word_max_length=25; sent_max_length=85; batch_limit_dev=999999999; dropout=0.7; 
python main.py \
  train \
  --signature_prefix "${tb_tag}.${word_max_length}.${sent_max_length}.ft.d${dropout}" \
  --wordvec_file "tmp/ft/cc.${lang}.300.vec.gz" \
  --train_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-train.conllu" \
  --dev_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-dev.conllu" \
  --test_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-test.conllu" \
  --save_dir "tmp/out" \
  --epochs 200 \
  --epochs_early_stopping 10 \
  --checkpoint_rolling True \
  --batch_size 1500 \
  --batch_per_epoch 400 \
  --batch_per_console_summary 50 \
  --batch_size_dev 1500 \
  --batch_limit_dev "${batch_limit_dev}" \
  --model_inputs char word \
  --model_word_dense_size None \
  --model_word_max_length "${word_max_length}" \
  --model_char_embedding_dim 200 \
  --model_char_conv_layers 3 \
  --model_char_conv_size 200 \
  --model_core_type transformer \
  \
  --model_core_transformer_input_dropout ${dropout} \
  --model_core_transformer_hidden_size 256 \
  --model_core_transformer_use_timing_signal True \
  --model_core_transformer_sent_max_length "${sent_max_length}" \
  --model_core_transformer_layers 3 \
  --model_core_transformer_attention_key_dense_size 128 \
  --model_core_transformer_attention_value_dense_size 128 \
  --model_core_transformer_attention_heads 16 \
  --model_core_transformer_attention_dropout ${dropout} \
  --model_core_transformer_pff_layers 2 \
  --model_core_transformer_pff_filter_size 1024 \
  --model_core_transformer_pff_dropout ${dropout} \
  --model_core_transformer_layer_dropout ${dropout} \
  \
  --model_core_bilstm_layers 2 \
  --model_core_bilstm_layer_size 512 \
  --model_core_bilstm_noise 0.0 \
  --model_core_bilstm_layer_dropout ${dropout} \
  --model_core_bilstm_dropout ${dropout} \
  \
  --model_head_dense_size 256 \
  --model_deprel_dense_size 50 \
  --model_upos_dense_size 50 \
  --model_feats_dense_size 512 \
  --model_lemma_dense_size 50 \
  --model_dropout ${dropout} \
  --model_noise 0.0
  #--optimizer_lr 0.01 \
  #--signature_suffix lr.01 \


lang='no'; tb_name='Norwegian-Bokmaal'; tb_tag='no_bokmaal'; word_max_length=75; sent_max_length=125; batch_limit_dev=999999999; dropout=0.7; 
python main.py \
  train \
  --signature_prefix "${tb_tag}.${word_max_length}.${sent_max_length}.ft.d${dropout}" \
  --wordvec_file "tmp/ft/cc.${lang}.300.vec.gz" \
  --train_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-train.conllu" \
  --dev_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-dev.conllu" \
  --test_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-test.conllu" \
  --save_dir "tmp/out" \
  --epochs 200 \
  --epochs_early_stopping 10 \
  --checkpoint_rolling True \
  --batch_size 1500 \
  --batch_per_epoch 400 \
  --batch_per_console_summary 50 \
  --batch_size_dev 1500 \
  --batch_limit_dev "${batch_limit_dev}" \
  --model_inputs char word \
  --model_word_dense_size None \
  --model_word_max_length "${word_max_length}" \
  --model_char_embedding_dim 150 \
  --model_char_conv_layers 3 \
  --model_char_conv_size 150 \
  --model_core_type transformer \
  \
  --model_core_transformer_input_dropout ${dropout} \
  --model_core_transformer_hidden_size 256 \
  --model_core_transformer_use_timing_signal True \
  --model_core_transformer_sent_max_length "${sent_max_length}" \
  --model_core_transformer_layers 3 \
  --model_core_transformer_attention_key_dense_size 128 \
  --model_core_transformer_attention_value_dense_size 128 \
  --model_core_transformer_attention_heads 16 \
  --model_core_transformer_attention_dropout ${dropout} \
  --model_core_transformer_pff_layers 2 \
  --model_core_transformer_pff_filter_size 1024 \
  --model_core_transformer_pff_dropout ${dropout} \
  --model_core_transformer_layer_dropout ${dropout} \
  \
  --model_core_bilstm_layers 2 \
  --model_core_bilstm_layer_size 512 \
  --model_core_bilstm_noise 0.0 \
  --model_core_bilstm_layer_dropout ${dropout} \
  --model_core_bilstm_dropout ${dropout} \
  \
  --model_head_dense_size 256 \
  --model_deprel_dense_size 50 \
  --model_upos_dense_size 50 \
  --model_feats_dense_size 512 \
  --model_lemma_dense_size 50 \
  --model_dropout ${dropout} \
  --model_noise 0.0
  #--optimizer_lr 0.01 \
  #--signature_suffix lr.01 \


lang='no'; tb_name='Norwegian-Nynorsk'; tb_tag='no_nynorsk'; word_max_length=75; sent_max_length=125; batch_limit_dev=999999999; dropout=0.7; 
python main.py \
  train \
  --signature_prefix "${tb_tag}.${word_max_length}.${sent_max_length}.ft.d${dropout}" \
  --wordvec_file "tmp/ft/cc.${lang}.300.vec.gz" \
  --train_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-train.conllu" \
  --dev_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-dev.conllu" \
  --test_file "tmp/ud-treebanks-v2.3/UD_${tb_name}/${tb_tag}-ud-test.conllu" \
  --save_dir "tmp/out" \
  --epochs 200 \
  --epochs_early_stopping 10 \
  --checkpoint_rolling True \
  --batch_size 1500 \
  --batch_per_epoch 400 \
  --batch_per_console_summary 50 \
  --batch_size_dev 1500 \
  --batch_limit_dev "${batch_limit_dev}" \
  --model_inputs char word \
  --model_word_dense_size None \
  --model_word_max_length "${word_max_length}" \
  --model_char_embedding_dim 150 \
  --model_char_conv_layers 3 \
  --model_char_conv_size 150 \
  --model_core_type transformer \
  \
  --model_core_transformer_input_dropout ${dropout} \
  --model_core_transformer_hidden_size 256 \
  --model_core_transformer_use_timing_signal True \
  --model_core_transformer_sent_max_length "${sent_max_length}" \
  --model_core_transformer_layers 3 \
  --model_core_transformer_attention_key_dense_size 128 \
  --model_core_transformer_attention_value_dense_size 128 \
  --model_core_transformer_attention_heads 16 \
  --model_core_transformer_attention_dropout ${dropout} \
  --model_core_transformer_pff_layers 2 \
  --model_core_transformer_pff_filter_size 1024 \
  --model_core_transformer_pff_dropout ${dropout} \
  --model_core_transformer_layer_dropout ${dropout} \
  \
  --model_core_bilstm_layers 2 \
  --model_core_bilstm_layer_size 512 \
  --model_core_bilstm_noise 0.0 \
  --model_core_bilstm_layer_dropout ${dropout} \
  --model_core_bilstm_dropout ${dropout} \
  \
  --model_head_dense_size 256 \
  --model_deprel_dense_size 50 \
  --model_upos_dense_size 50 \
  --model_feats_dense_size 512 \
  --model_lemma_dense_size 50 \
  --model_dropout ${dropout} \
  --model_noise 0.0
  #--optimizer_lr 0.01 \
  #--signature_suffix lr.01 \

