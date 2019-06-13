#!/usr/bin/env bash
export GLUE_DIR=/data/users/maxinyu/pytorch-pretrained-BERT/data/glue
export TASK_NAME=msmarco

for ((i=12; i>6 ;i--)); do python prob_bert.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --freeze_bert \
  --layer_index_to_prob $i \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --bert_model output/msmarco \
  --max_seq_length 128 \
  --train_batch_size 80 \
  --eval_batch_size 100 \
  --learning_rate 1e-6 \
  --num_train_epochs 5.0 \
  --patience 1 \
  --mask_cls \
  --threshold 0.005 \
  --output_dir output/${TASK_NAME}_prob_original_mask_cls_lr1_$i/ ; done
