#!/usr/bin/env bash

for split in 'train' 'dev' 'test'; do python preprocess_ewt_pos_tagging.py \
  --input_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/pos/en_ewt-ud-${split}.conllu \
  --output_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/pos/truncated_en_ewt-ud-${split}.conllu \
  --bert_model bert-base-cased \
  --max_seq_length 126 ; done
