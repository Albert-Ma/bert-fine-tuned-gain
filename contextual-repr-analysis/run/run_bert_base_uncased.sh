#!/usr/bin/env bash

for ((i=8; i<10; i++)); do allennlp train configs/bert_base_uncased/adposition_supersense_tagging_function.json \
    -s models/bert_base_uncased/ps_fun/msmarco/ps_layer_${i} \
    --include-package contexteval \
    --overrides '{"dataset_reader": {"contextualizer": {"layer_num": '${i}'}}, "validation_dataset_reader": {"contextualizer": {"layer_num": '${i}'}}}'; done
