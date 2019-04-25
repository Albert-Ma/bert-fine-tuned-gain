#!/usr/bin/env bash

for i in 0 1; do allennlp train experiment_configs/bert_base_cased/ewt_pos_tagging.json \
    -s models/bert_base_cased/pos/ewt_pos_tagging_layer_${i} --include-package contexteval \
    --overrides '{"dataset_reader": {"contextualizer": {"layer_num": '${i}'}}, "validation_dataset_reader": {"contextualizer": {"layer_num": '${i}'}}}'; done