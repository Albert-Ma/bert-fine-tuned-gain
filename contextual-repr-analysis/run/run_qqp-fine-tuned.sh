#!/usr/bin/env bash

for ((i=0; i<12; i++)); do allennlp train configs/qqp-fine-tuned/adposition_supersense_tagging_function.json \
    -s models/qqp/ps_fun/ps_layer_${i} --include-package contexteval \
    --overrides '{"dataset_reader": {"contextualizer": {"layer_num": '${i}'}}, "validation_dataset_reader": {"contextualizer": {"layer_num": '${i}'}}}'; done