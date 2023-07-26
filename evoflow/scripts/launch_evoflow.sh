#!/bin/bash

declare -a datasets=("breastcancer" "glass" "hillvalley" "ionosphere" "spambase" "winequalityred" "abalone" "amazon" "car" "convex" "dexter" "dorothea" "germancredit" "gisette" "krvskp" "madelon" "secom" "semeion" "shuttle" "waveform" "winequalitywhite" "yeast")

seed=180212

for data in "${datasets[@]}"
do
    python examples/exp_params.py $data $seed configs/evoflow_1h.py
done
