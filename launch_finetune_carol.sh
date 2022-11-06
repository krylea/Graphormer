#!/usr/bin/env bash

MODELS=("base" "paw" "properties" "subshell")
dataset=$1

for model in "${MODELS[@]}"
do
    sbatch finetune_carol.sh "finetune_carol_${dataset}_${model}" $model $dataset
done