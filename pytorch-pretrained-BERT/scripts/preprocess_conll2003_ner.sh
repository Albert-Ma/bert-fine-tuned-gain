#!/usr/bin/env bash

for split in 'train' 'dev' 'test'; do python preprocess_conll2003.py \
  --input_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/chunk/${split}.txt \
  --output_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/chunk/truncated_${split}.txt \
  --bert_model bert-base-cased \
  --max_seq_length 126 ; done