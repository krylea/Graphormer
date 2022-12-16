#!/usr/bin/env bash

MODELS=("base" "paw" "properties" "subshell")
dataset=$1
prefix=$2
folds=8


for (( i=0; i<$folds; i++ ))
do
    for model in "${MODELS[@]}"
    do
        folder="$prefix_${dataset}/${model}_${i}"
        if [ ! -a "ckpts/${folder}/checkpoint_last.pt" ]
        then
            sbatch useful_scripts/finetune_carol.sh $folder $model "${dataset}/kfold/${i}"
        fi
    done

    #sbatch useful_scripts/launch_carol.sh "$prefix_${dataset}/nofinetune_${i}" "${dataset}/kfold/${i}"
done