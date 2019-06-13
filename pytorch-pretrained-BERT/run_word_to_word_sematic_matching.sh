#!/usr/bin/env bash
# truncated_en_ewt-ud_sentences.txt
# --layers=-5
python run_word_to_word_semantic_matching.py \
  --input_file truncated_ratings.txt \
  --task_name scws \
  --bert_model output/bert-base-uncased \
  --max_seq_length 128 \
  --batch_size 32 \
  --use_sentence_b
