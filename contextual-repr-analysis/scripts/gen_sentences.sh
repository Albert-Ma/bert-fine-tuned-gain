#!/usr/bin/env bash
python scripts/get_config_sentences.py \
  --config-path experiment_configs/bert_base_cased/ewt_pos_tagging.json \
  --output-path data/pos/truncated_en_ewt-ud_sentences.txt