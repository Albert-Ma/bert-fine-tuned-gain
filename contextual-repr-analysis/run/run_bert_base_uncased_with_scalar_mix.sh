#!/usr/bin/env bash

allennlp train configs/bert_base_uncased_trained_scalar_mix/adposition_supersense_tagging_function.json \
    -s models/bert_base_uncased/ps_fun/qnli/scalar_mix \
    --include-package contexteval