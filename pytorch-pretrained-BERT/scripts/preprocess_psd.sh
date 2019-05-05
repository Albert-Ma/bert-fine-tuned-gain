#!/usr/bin/env bash

for split in 'train' 'dev' 'test'; do python preprocess_psd.py \
  --input_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/psd/streusle.ud_${split}.json \
  --output_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/psd/truncated_streusle.ud_${split}.json \
  --bert_model bert-base-cased \
  --max_seq_length 126 ; done