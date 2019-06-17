#!/usr/bin/env bash

allennlp train configs/qqp_trained_scalar_mix/adposition_supersense_tagging_function.json \
    -s models/qqp/ps_fun/scalar_mix/ \
    --include-package contexteval