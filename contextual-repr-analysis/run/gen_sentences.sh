#!/usr/bin/env bash
export TASK='qnli'
export DATA='pos'

python run/get_config_sentences.py \
  --config-path configs/bert_base_uncased/ontonotes_pos.json \
  --output-path data/${DATA}/truncated_onto_sentences.txt