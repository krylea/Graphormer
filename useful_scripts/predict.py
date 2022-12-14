import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks, criterions
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, r2_score

import copy
import sys
from os import path

#sys.path.append( path.dirname( path.abspath(os.getcwd()) ) ) 
#import graphormer
#import graphormer.tasks.is2re
#from graphormer.pretrain import load_pretrained_model

import logging
import argparse
import omegaconf

import matplotlib.pyplot as plt

def convert_omegaconf(cfg):
    from omegaconf import __version__ as oc_version
    from omegaconf import _utils
    from omegaconf import OmegaConf

    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True

    cfg = OmegaConf.create(cfg)

    _utils.is_primitive_type = old_primitive
    OmegaConf.set_struct(cfg, True)
    return cfg

def load_checkpoint(checkpoint_path, criterion_name="mae_deltapos", user_dir='./graphormer'):
    checkpoint = torch.load(checkpoint_path)
    cfg = checkpoint['cfg']

    cfg = convert_omegaconf(cfg)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    # initialize task
    cfg.task.user_dir=user_dir
    utils.import_user_module(cfg.task)
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    # load checkpoint
    model_state = checkpoint["model"]
    model.load_state_dict(
        model_state, strict=True, model_cfg=cfg.model
    )
    model.to(torch.cuda.current_device())
    criterion = criterions.CRITERION_REGISTRY[criterion_name]
    return cfg, task, model, criterion

def eval_split(cfg, task, model, criterion, split, max_steps=-1, dataset=None):
    def target_transform(target, criterion):
        return (target - criterion.e_mean) / criterion.e_std
    def inv_transform(pred, criterion):
        return pred*criterion.e_std + criterion.e_mean
    
    #utils.import_user_module(cfg.common)
    
    if dataset is not None:
        cfg['task'].data = dataset
    task = tasks.setup_task(cfg.task)
    task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )
    # infer
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample)
            y = model(**sample["net_input"])[0].reshape(-1)
            #y_pred.extend(y.detach().cpu())
            #y_true.extend(target_transform(sample["targets"]["relaxed_energy"], criterion).detach().cpu().reshape(-1)[:y.shape[0]])
            y_pred.extend(inv_transform(y, criterion).detach().cpu())
            y_true.extend(sample["targets"]["relaxed_energy"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()
            if max_steps > 0 and i / cfg.task.batch_size_valid >= max_steps:
                break
    # save predictions
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)
    return y_pred, y_true

def parity_plot(splits, title=None, bounds=[-8,0,-8,0]):
    plt.figure(figsize=(8,6))
    plt.axis(bounds)
    if len(splits) > 1:
        for split_dict in splits:
            label="%s\nMAE:%.3f" % (split_dict['split'], split_dict['mae'])
            plt.scatter(split_dict['y_true'], split_dict['y_pred'], label=label)
    else:
        for split_dict in splits:
            label="MAE:%.3f" % (split_dict['mae'])
            plt.scatter(split_dict['y_true'], split_dict['y_pred'], label=label)
    plt.plot(np.linspace(bounds[0], bounds[1], 100), np.linspace(bounds[2], bounds[3], 100), 'k--')
    plt.title(title)
    plt.legend()
    plt.xlabel("E_ads DFT (eV)")
    plt.ylabel("E_ads NN (eV)")
    plt.show()

def get_run_results(run_name, dataset, checkpoint_dir="ckpts", splits=["train", "val", "test"], dataset_dir="/h/carolyu/kira_code/lmdbs"):
    cfg, task, model, criterion = load_checkpoint(os.path.join(checkpoint_dir, run_name, "checkpoint_last.pt"))

    data_dir = os.path.join(dataset_dir, dataset)
    print("%s Results:" % run_name)
    for split in splits:
        if not os.path.exists(data_dir, split, "data.lmdb"):
            continue
        else:
            y_pred, y_true = eval_split(cfg, task, model, criterion, split, dataset=data_dir)
            error = (y_pred - y_true).abs().mean().item()
            print("\t%s: %.4f" % (split, error))

if __name__ == '__main__':
    import argparse
    
    parser=argparse.ArgumentParser()
    parser.add_argument('run_names', type=str, nargs='+')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_dir', type=str, default='/h/carolyu/kira_code/lmdbs')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts')

    args = parser.parse_args()

    for run_name in args.run_names:
        get_run_results(run_name, args.dataset)

    
