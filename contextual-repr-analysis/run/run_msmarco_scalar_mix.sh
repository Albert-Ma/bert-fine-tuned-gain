#!/usr/bin/env bash

allennlp train configs/msmarco_trained_scalar_mix/adposition_supersense_tagging_function.json \
    -s models/msmarco/b_ps_fun/scalar_mix/ \
    --include-package contexteval