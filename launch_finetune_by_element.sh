#!/usr/bin/env bash

ELEMENTS=(17 37 55 75 81)
SPLITS=(50 200 1000 5000)

for e in "${ELEMENTS[@]}"
do
    for s in "${SPLITS[@]}"
    do
        sbatch finetune_oc20_by_element.sh "finetune_$e_$s" $e $s
    done
done