#!/usr/bin/env bash

MODELS=("base" "paw" "properties" "subshell")
dataset=$1
folds=8

for (( i=0; i<$folds; i++ ))
do
    for model in "${MODELS[@]}"
    do
        sbatch finetune_carol.sh "kfold_${dataset}/${model}_${i}" $model "${dataset}/kfold/${i}"
    done

    sbatch launch_oc20.sh "kfold_${dataset}/nofinetune_${i}" "${dataset}/kfold/${i}"
done