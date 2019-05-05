#!/usr/bin/env bash
# 'train' 'dev' 'test'
for split in 'train' 'dev' 'test'; do python preprocess_ged.py \
  --input_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/gec/fce-public.${split} \
  --output_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/gec/truncated_fce-public.${split} \
  --bert_model bert-base-cased \
  --max_seq_length 126 ; done
