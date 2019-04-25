#!/usr/bin/env bash
# truncated_en_ewt-ud_sentences.txt
python extract_features.py \
  --input_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/pos/truncated_en_ewt-ud_sentences.txt \
  --output_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/contextualizers/bert_base_cased/pos/ewt_pos.hdf5 \
  --bert_model bert-base-cased \
  --max_seq_length 128 \
  --batch_size 32 \
  --layers=-5