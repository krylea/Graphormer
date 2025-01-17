#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#SBATCH --job-name=test
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --partition=t4v1,t4v2,p100,rtx6000
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=25GB

run_name=$1
model=$2
datapath=$3

if [ $model == "base" ]
then
    modelpath="base_by_element"
    arch="graphormer3d_base"
    argstring=""
elif [ $model == "paw" ]
then
    modelpath="subatom_by_element"
    arch="graphormer3dsubatom_base"
    argstring="--atom-embed-path element_tensors/paw.pt"
elif [ $model == "properties" ]
then
    modelpath="subatom_properties_by_element"
    arch="graphormer3dsubatom_base"
    argstring="--atom-embed-path element_tensors/properties.pt"
elif [ $model == "subshell" ]
then
    modelpath="test_subshell"
    arch="graphormer3dsubshell_base"
    argstring="--core-embed-path element_tensors/core_configs.pt --valence-embed-path element_tensors/valence_configs.pt"
fi

lr=${lr:-2e-5}
warmup_steps=${warmup_steps:-0}
total_steps=${total_steps:-5000}
layers=${layers:-12}
hidden_size=${hidden_size:-768}
num_head=${num_head:-48}
batch_size=${batch_size:-2}
seed=${seed:-1}
clip_norm=${clip_norm:-5}
blocks=${blocks:-4}
node_loss_weight=${node_loss_weight:-15}
update_freq=${update_freq:-1}

save_dir=./ckpts/${run_name}
tsb_dir=./tsbs/${run_name}
mkdir -p $save_dir

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "batch_size: ${batch_size}"
echo "layers: ${layers}"
echo "update_freq: ${update_freq}"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "total_steps: ${total_steps}"
echo "clip_norm: ${clip_norm}"
echo "blocks: ${blocks}"
echo "node_loss_weight: ${node_loss_weight}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "==============================================================================="

fairseq-train --user-dir ./graphormer  \
       /scratch/hdd001/home/$USER/carol-lmdbs/$datapath --valid-subset val --best-checkpoint-metric loss \
       --num-workers 0 --ddp-backend=c10d \
       --task is2re --criterion mae_deltapos --arch $arch  \
       --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm $clip_norm \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $total_steps --batch-size $batch_size \
       --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $tsb_dir \
       --embed-dim $hidden_size --ffn-embed-dim $hidden_size --attention-heads $num_head \
       --max-update $total_steps --log-interval 100 --log-format simple \
       --save-interval-updates 500 --validate-interval-updates 250 --keep-interval-updates 30 --no-epoch-checkpoints  \
       --save-dir $save_dir --layers $layers --blocks $blocks --required-batch-size-multiple 1  --node-loss-weight $node_loss_weight \
       --finetune-from-model ./ckpts/$modelpath/checkpoint_last.pt $argstring
