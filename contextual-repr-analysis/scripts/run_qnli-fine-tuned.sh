#!/usr/bin/env bash

for ((i=0; i<12; i++)); do allennlp train experiment_configs/qnli-fine-tuned/coreference_arc_prediction.json \
    -s models/qnli/cap/cap_layer_${i} --include-package contexteval \
    --overrides '{"dataset_reader": {"contextualizer": {"layer_num": '${i}'}}, "validation_dataset_reader": {"contextualizer": {"layer_num": '${i}'}}}'; done