
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
    def __init__(self, embed_dim, n_max, l_max, atom_configs, occupancy_correction=False, mode='train'):
        super().__init__()
        self.embed_dim = embed_dim
        augmented_configs = torch.cat([torch.zeros(1,n_max*l_max), atom_configs], dim=0)
        self.register_buffer('atom_configs', augmented_configs)   # Nele x nmax*lmax
        self.subshell_embeds = nn.Parameter(torch.empty(n_max * l_max, embed_dim))
        self.occupancy_correction = occupancy_correction
        if occupancy_correction:
            self.occupancy_embeds = nn.Parameter(torch.empty(n_max * l_max, embed_dim))

        self.mode = mode
        self.corrections = None

        self.init_params()
    
    def init_params(self):
        nn.init.normal_(self.subshell_embeds)
        if self.occupancy_correction:
            nn.init.normal_(self.occupancy_embeds)

    def set_finetune(self, n_atom):
        self.corrections = nn.Embedding(n_atom+1, self.embed_dim, padding_idx=0)
        self.subshell_embeds.requires_grad_(False)
        if self.occupancy_correction:
            self.occupancy_embeds.requires_grad_(False)
        self.mode='finetune'

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
        
        if self.corrections is not None:
            # could also just save fixed vectors for all atoms once training is done so we dont have to recalculate the above every time, 
            # but idk if the overhead actually matters
            corrections = self.corrections[atom_indices]
            atom_vectors += corrections

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
        self.mode='train'

    def set_finetune(self, n_atom):
        self.mode = 'finetune'
        self.valence_embeds.set_finetune(n_atom)
        self.core_embeds.set_finetune(n_atom)
    
    def forward(self, atom_indices):
        valence_vectors = self.valence_embeds(atom_indices)
        core_vectors = self.core_embeds(atom_indices)
        return torch.cat([valence_vectors, core_vectors], dim=-1)


import math

class RBFSubatom(nn.Module):
    def __init__(self, K, embed_dim, hidden_dim):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))

        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.temps, 0.1, 10)
        #nn.init.constant_(self.bias.weight, 0)
        #nn.init.constant_(self.mul.weight, 1)

        self.mode = 'train'

    def set_finetune(self, n_atom):
        self.n_atom = n_atom
        self.mul_correction = nn.Embedding(n_atom*n_atom, 1)
        self.bias_correction = nn.Embedding(n_atom*n_atom, 1)
        nn.init.constant_(self.bias_correction.weight, 0)
        nn.init.constant_(self.mul_correction.weight, 0)
        self.lambda_mul = nn.Parameter(torch.tensor([1.]))
        self.lambda_bias = nn.Parameter(torch.tensor([1.]))
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.mode='finetune'

    def forward(self, x, atom_embeds, atom_indices=None):
        n_graph, n_node = atom_embeds.size()[:2]
        edge_features = self.net(
            torch.cat([
                atom_embeds.view(n_graph, n_node, 1, self.embed_dim).expand(n_graph, n_node, n_node, self.embed_dim), 
                atom_embeds.view(n_graph, 1, n_node, self.embed_dim).expand(n_graph, n_node, n_node, self.embed_dim)
            ], dim=-1)
        ) / math.sqrt(self.embed_dim * 2)
        mul, bias = edge_features.chunk(2, dim=-1)

        if self.mul_correction is not None and self.bias_correction is not None and atom_indices is not None:
            edge_types = atom_indices.view(n_graph, n_node, 1) * self.atom_types + atom_indices.view(
                n_graph, 1, n_node
            )
            mul_corr = self.mul_correction(edge_types)
            bias_corr = self.bias_correction(edge_types)
            mul = self.lambda_mul * mul + (1-self.lambda_mul) * mul_corr
            bias = self.lambda_bias * bias + (1-self.lambda_bias) * bias_corr

        x = mul * x.unsqueeze(-1) + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)
