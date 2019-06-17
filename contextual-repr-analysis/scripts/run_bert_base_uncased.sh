#!/usr/bin/env bash

for ((i=0; i<12; i++)); do allennlp train experiment_configs/bert_base_cased/coreference_arc_prediction.json \
    -s models/bert_base_cased/cap/cap_layer_${i} --include-package contexteval \
    --overrides '{"dataset_reader": {"contextualizer": {"layer_num": '${i}'}}, "validation_dataset_reader": {"contextualizer": {"layer_num": '${i}'}}}'; done