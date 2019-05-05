#!/usr/bin/env bash
# 'train' 'dev' 'test'
for split in 'train' 'dev' 'test'; do python preprocess_onto.py \
  --input_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/cf/${split}.english.v4_gold_conll \
  --output_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/cf/truncated_${split}.english.v4_gold_conll \
  --bert_model bert-base-uncased \
  --max_seq_length 2028 ; done
