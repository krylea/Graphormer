# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Sequence, Union, Dict

import pickle
from functools import lru_cache

import lmdb

import numpy as np
import torch
from torch import Tensor
from fairseq.data import (
    FairseqDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task

from ..data.dataset import EpochShuffleDataset

class LMDBDataset:
    def __init__(self, db_path):
        super().__init__()
        assert Path(db_path).exists(), f"{db_path}: No such file or directory"
        self.env = lmdb.Environment(
            db_path,
            map_size=(1024 ** 3) * 256,
            subdir=False,
            readonly=True,
            readahead=True,
            meminit=False,
            lock=False
        )
        self.len: int = self.env.stat()["entries"]

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, float]]:
        if idx < 0 or idx >= self.len:
            raise IndexError
        data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
        return dict(
            pos=torch.as_tensor(data["pos"]).float(),
            pos_relaxed=torch.as_tensor(data["pos_relaxed"]).float(),
            cell=torch.as_tensor(data["cell"]).float().view(3, 3),
            atoms=torch.as_tensor(data["atomic_numbers"]).long(),
            tags=torch.as_tensor(data["tags"]).long(),
            relaxed_energy=data["y_relaxed"],  # python float
        )


class PBCDataset:
    def __init__(self, dataset: LMDBDataset):
        self.dataset = dataset
        self.cell_offsets = torch.tensor(
            [
                [-1, -1, 0],
                [-1, 0, 0],
                [-1, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [1, -1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
        ).float()
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = 8
        self.filter_by_tag = True

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]

        pos = data["pos"]
        pos_relaxed = data["pos_relaxed"]
        cell = data["cell"]
        atoms = data["atoms"]
        tags = data["tags"]

        offsets = torch.matmul(self.cell_offsets, cell).view(self.n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        expand_pos_relaxed = (
            pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets
        ).view(-1, 3)
        src_pos = pos[tags > 1] if self.filter_by_tag else pos

        dist: Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
        used_mask = (dist < self.cutoff).any(dim=0) & tags.ne(2).repeat(
            self.n_cells
        )  # not copy ads
        used_expand_pos = expand_pos[used_mask]
        used_expand_pos_relaxed = expand_pos_relaxed[used_mask]

        used_expand_tags = tags.repeat(self.n_cells)[
            used_mask
        ]  # original implementation use zeros, need to test
        return dict(
            pos=torch.cat([pos, used_expand_pos], dim=0),
            atoms=torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]]),
            tags=torch.cat([tags, used_expand_tags]),
            real_mask=torch.cat(
                [
                    torch.ones_like(tags, dtype=torch.bool),
                    torch.zeros_like(used_expand_tags, dtype=torch.bool),
                ]
            ),
            deltapos=torch.cat(
                [pos_relaxed - pos, used_expand_pos_relaxed - used_expand_pos], dim=0
            ),
            relaxed_energy=data["relaxed_energy"],
        )


def pad_1d(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]), fill, dtype=samples[0].dtype
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out


class AtomDataset(FairseqDataset):
    def __init__(self, dataset, keyword, add_atoms=()):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.atom_list = [
            1,
            5,
            6,
            7,
            8,
            11,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            55,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
        ]
        # fill others as unk
        unk_idx = len(self.atom_list) + 1
        self.atom_mapper = torch.full((128,), unk_idx)
        for idx, atom in enumerate(self.atom_list):
            self.atom_mapper[atom] = idx + 1  # reserve 0 for paddin
        
        # note: this solution only works if there are less than (63-unk_idx) atoms being added
        # also: maybe re-initialize embeddings for new atoms? not sure if matters or not
        if len(add_atoms) > 0:
            self.add_atoms = add_atoms
            for i, atom_idx in enumerate(add_atoms):
                self.atom_list.append(atom_idx)
                self.atom_mapper[atom_idx] = unk_idx + 1 + i


    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        atoms: Tensor = self.dataset[index][self.keyword]
        return self.atom_mapper[atoms]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return pad_1d(samples)


class KeywordDataset(FairseqDataset):
    def __init__(self, dataset, keyword, is_scalar=False, pad_fill=0):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.is_scalar = is_scalar
        self.pad_fill = pad_fill

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index][self.keyword]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.is_scalar:
            return torch.tensor(samples)
        return pad_1d(samples, fill=self.pad_fill)


@register_task("is2re")
class IS2RETask(FairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", metavar="FILE", help="directory for data")
        parser.add_argument("--add_atoms", nargs='+', type=int, default=(), help="additional atoms to add")
        parser.add_argument("--freeze", action='store_true', help="freeze weights")

    @property
    def target_dictionary(self):
        return None

    def load_dataset(self, split, combine=False, **kwargs):
        '''
        assert split in [
            "train",
            "val_id",
            "val_ood_ads",
            "val_ood_cat",
            "val_ood_both",
            "test_id",
            "test_ood_ads",
            "test_ood_cat",
            "test_ood_both",
        ], "invalid split: {}!".format(split)
        '''
        print(" > Loading {} ...".format(split))

        #db_path = str(Path(self.cfg.data) / f"{split}.lmdb")
        db_path = str(Path(self.cfg.data) / split / "data.lmdb")
        lmdb_dataset = LMDBDataset(db_path)
        pbc_dataset = PBCDataset(lmdb_dataset)

        atoms = AtomDataset(pbc_dataset, "atoms", add_atoms=self.cfg.add_atoms)
        tags = KeywordDataset(pbc_dataset, "tags")
        real_mask = KeywordDataset(pbc_dataset, "real_mask")

        pos = KeywordDataset(pbc_dataset, "pos")

        relaxed_energy = KeywordDataset(pbc_dataset, "relaxed_energy", is_scalar=True)
        deltapos = KeywordDataset(pbc_dataset, "deltapos")

        dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pos": pos,
                    "atoms": atoms,
                    "tags": tags,
                    "real_mask": real_mask,
                },
                "targets": {
                    "relaxed_energy": relaxed_energy,
                    "deltapos": deltapos,
                },
            },
            sizes=[np.zeros(len(atoms))],
        )

        if split == "train":
            dataset = EpochShuffleDataset(
                dataset,
                num_samples=len(atoms),
                seed=self.cfg.seed,
            )

        print("| Loaded {} with {} samples".format(split, len(dataset)))
        self.datasets[split] = dataset

    def build_model(self, args):
        model = super().build_model(args)

        if len(args.add_atoms) > 0 and args.freeze:
            mask = torch.ones(64, args.embed_dim)
            unk_idx = 57
            for i, _ in enumerate(args.add_atoms):
                mask[unk_idx+i+1, :] = 0
            mask = mask.bool()
            def hook_fn(grad):
                out = grad.clone()
                out[mask] = 0
                return out
            
            for p in model.parameters():
                p.requires_grad = False
            model.atom_encoder.weight.requires_grad = True
            model.atom_encoder.weight.register_hook(hook_fn)

        return model

