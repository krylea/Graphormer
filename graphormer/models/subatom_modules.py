
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
    def __init__(self, embed_dim, n_max, l_max, atom_configs):
        self.embed_dim = embed_dim
        self.atom_configs = atom_configs    # Natom x 
        self.subshell_embeds = nn.Parameter(torch.empty(n_max, l_max, embed_dim))
        self.occupancy_embeds = nn.Parameter(torch.empty(n_max, l_max, embed_dim))

        self.init_params()
    
    def init_params(self):
        return

    def forward(self, atom_indices):
        atom_vectors = 
