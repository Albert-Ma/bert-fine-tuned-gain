#!/usr/bin/env bash
export GLUE_DIR=/data/users/maxinyu/pytorch-pretrained-BERT/data/glue
export TASK_NAME=cvt-qnli

python cvt_classifier.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --bert_model output/bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --eval_batch_size 100 \
  --learning_rate 2e-5 \
  --num_train_epochs 30.0 \
  --patience 3 \
  --mask_cls \
  --threshold 0.001 \
  --output_dir output/cvt_${TASK_NAME}/
