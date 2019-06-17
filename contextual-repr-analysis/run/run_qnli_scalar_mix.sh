#!/usr/bin/env bash

allennlp train configs/qnli_trained_scalar_mix/adposition_supersense_tagging_function.json \
    -s models/qnli/ps_fun/scalar_mix/ \
    --include-package contexteval