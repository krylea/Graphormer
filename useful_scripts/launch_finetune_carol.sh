#!/usr/bin/env bash

MODELS=("base" "paw" "properties" "subshell")
dataset=$1
prefix=$2

for model in "${MODELS[@]}"
do
    sbatch useful_scripts/finetune_carol.sh "${prefix}_${dataset}_${model}" $model $dataset
done

sbatch useful_scripts/train_carol.sh "${prefix}_${dataset}_nofinetune" $dataset