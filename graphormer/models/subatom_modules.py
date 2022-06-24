
from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F

class FixedAtomEmbedding(nn.Module):
    @classmethod
    def from_file(cls, tensor_file):
        embed_weight = torch.load(tensor_file)
        return cls(embed_weight)

    def __init__(self, atom_tensors, padding_idx=0):
        super().__init__()
        self.atom_base_dim = atom_tensors.size(1)
        augmented_embed_weight = torch.cat([torch.zeros(1,self.atom_base_dim), atom_tensors], dim=0)
        self.register_buffer('embed', augmented_embed_weight)
        self.padding_idx=padding_idx

        #self.embed = nn.Embedding.from_pretrained(augmented_embed_weight, freeze=True, padding_idx=0)

    def forward(self, indices):
        return F.embedding(indices, self.embed, self.padding_idx, None, 2, False, False)


class SubshellEmbedding(nn.Module):
    def __init__(self, embed_dim, n_max, l_max, atom_configs, occupancy_correction=False):
        super().__init__()
        self.embed_dim = embed_dim
        augmented_configs = torch.cat([torch.zeros(1,n_max*l_max), atom_configs], dim=0)
        self.register_buffer('atom_configs', augmented_configs)   # Nele x nmax*lmax
        self.subshell_embeds = nn.Parameter(torch.empty(n_max * l_max, embed_dim))
        self.occupancy_correction = occupancy_correction
        if occupancy_correction:
            self.occupancy_embeds = nn.Parameter(torch.empty(n_max * l_max, embed_dim))

        self.init_params()
    
    def init_params(self):
        nn.init.normal_(self.subshell_embeds)
        if self.occupancy_correction:
            nn.init.normal_(self.occupancy_embeds)

    def forward(self, atom_indices):
        atom_repr = self.atom_configs[atom_indices] # bs x n_atom x nmax*lmax
        if not self.occupancy_correction:
            atom_vectors = (atom_repr.unsqueeze(-1) * self.subshell_embeds.view(1, 1, *self.subshell_embeds.size())).sum(dim=-2)
        else:
            base_atom_repr = (atom_repr != 0).long()
            occupancy_repr = torch.max(atom_repr-1, torch.zeros_like(atom_repr))
            base_atom_vectors = (base_atom_repr.unsqueeze(-1) * self.subshell_embeds.view(1, 1, *self.subshell_embeds.size())).sum(dim=-2)
            occupancy_vectors = (occupancy_repr.unsqueeze(-1) * self.occupancy_embeds.view(1, 1, *self.occupancy_embeds.size())).sum(dim=-2)
            atom_vectors = base_atom_vectors + occupancy_vectors
        return atom_vectors

class SubshellValenceEmbedding(nn.Module):
    @classmethod
    def from_file(cls, valence_file, core_file, embed_dim, hidden_dim, n_max, l_max, occupancy_correction):
        valence_configs = torch.load(valence_file)
        core_configs = torch.load(core_file)
        return cls(embed_dim, hidden_dim, n_max, l_max, valence_configs, core_configs, occupancy_correction)

    def __init__(self, embed_dim, hidden_dim, n_max, l_max, valence_configs, core_configs, occupancy_correction=False):
        super().__init__()
        self.valence_embeds = SubshellEmbedding(embed_dim, n_max, l_max, valence_configs, occupancy_correction)
        self.core_embeds = SubshellEmbedding(embed_dim, n_max, l_max, core_configs, occupancy_correction)
        #self.merge = nn.Sequential(
        #    nn.Linear(embed_dim*2, hidden_dim),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim, embed_dim)
        #)
    
    def forward(self, atom_indices):
        valence_vectors = self.valence_embeds(atom_indices)
        core_vectors = self.core_embeds(atom_indices)
        return torch.cat([valence_vectors, core_vectors], dim=-1)

