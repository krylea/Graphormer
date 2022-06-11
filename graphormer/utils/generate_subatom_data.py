

import torch
import torch.nn.functional as F

# adapted from https://github.com/txie-93/cgcnn
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return torch.exp(-(distances.unsqueeze(-1) - self.filter)**2 /
                      self.var**2)



class Bucketizer():
    def __init__(self, n_buckets, min_range, max_range):
        self.min_range = min_range
        self.max_range = max_range
        self.n_buckets = n_buckets
    
    def __call__(self, values):
        buckets = torch.floor((values - self.min_range)/self.n_buckets)
        return F.one_hot(buckets, num_classes=self.n_buckets)

def discrete_bucketizer(num_classes, min_index=0):
    return lambda x: F.onehot(x-min_index, num_classes=num_classes)


# expects dict of {propertyname: [property value for each element in order]} and {propertyname: bucketizer}
def bucketize_data(element_data, bucketizers):
    sorted_keys = sorted(element_data.keys())
    n_elements = len(element_data.values().__next__())
    vector_dim = sum([bucketizers[propertyname].n_buckets for propertyname in sorted_keys])
    tensor = torch.zeros(n_elements, vector_dim)
    n=0
    for propertyname in sorted_keys:
        element_vals = torch.tensor(element_data[propertyname])
        bucketized_vals = bucketizers[propertyname](element_vals)
        dn = bucketized_vals.size(-1)
        tensor[:, n:n+dn] = bucketized_vals
        n += dn
    return tensor


BUCKETIZERS = {
    'group': discrete_bucketizer(18, min_index=1),
    'period': discrete_bucketizer(9, min_index=1),
    'electronegativity': Bucketizer(10, 0.5, 4.0),
    'covalent_radius': Bucketizer(10, 25, 250),
    'valence_electrons': discrete_bucketizer(12, min_index=1),
    'first_ion_eng': Bucketizer(10, 1.3, 3.3),
    'electron_affinity': Bucketizer(10, -3, 3.7),
    'block': discrete_bucketizer(4, min_index=1),
    'atomic_volume': Bucketizer(10, 1.5, 4.3) 
}