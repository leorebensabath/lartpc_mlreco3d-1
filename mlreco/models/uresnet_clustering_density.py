import torch
import torch.nn as nn
import numpy as np
from scipy.special import gamma
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn
from collections import defaultdict
from sklearn.neighbors import RadiusNeighborsClassifier

from mlreco.models.uresnet_clustering_alt import UResNet, DiscriminativeLoss
from mlreco.models.ppn import PPN, PPNLoss


class ClusterNet_Density(nn.Module):
    """
    Run UResNet and use its encoding/decoding feature maps for PPN layers
    """

    def __init__(self, model_config):
        super(ClusterNet_Density, self).__init__()
        self.clusternet = UResNet(model_config)
        self._N_density = model_config.get('N_density', 5)
        self._num_radii = model_config.get('num_radii', 1)
        self._radii = model_config.get('radii', [1.0])
        self._dimension = model_config.get('data_dim', 3)
        m = model_config.get('filters', 16)
        leakiness = 0.2

        self.density_layers = scn.Sequential()

        def block(m, a, b):  # ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, b, b, 3, False)))
             ).add(scn.AddTable())

        for _ in range(self._N_density):
            block(self.density_layers, m, m)
        self.linear_densityMap = nn.Linear(m, self._num_radii)


    def forward(self, input):
        out = self.clusternet.forward(input)
        x_seg = out[0][0]
        embeddings = out[1][0]
        dmap = self.density_layers(embeddings[-1])
        dmap = self.output(dmap)
        dmap = self.linear_densityMap(dmap)

        return [[x_seg], 
                [embeddings],
                [dmap]]


class DensityLoss(nn.Module):
    """
    Enhancement of Discriminative Loss with Density Maps
    """
    def __init__(self, cfg, reduction='sum'):
        super(DensityLoss, self).__init__()
        self.clustering_loss = DiscriminativeLoss(cfg)
        self._cfg = cfg['modules']['discriminative_loss']
        self._model_config = cfg['modules']['uresnet_clustering']
        self._min_points = self._cfg.get('min_points', 10)
        self._radii = self._cfg.get('radii', [1.0])
        def compute_nsphere_volume(radii):
            n, l = self._model_config['hypDim'], []
            for r in radii:
                vol = np.pi**(n / 2.0) * r**2 / gamma(n / 2.0)
                l.append(vol)
            return l
        self._nsVols = torch.Tensor(compute_nsphere_volume(self._radii))
        self.bceloss = torch.nn.BCELoss()
        self.density_mode = self._cfg.get('density_mode', 0)

    def similarity_matrix(self, mat, mat2=None):
        """
        Computes the similarity matrix for pairwise distance computations.

        INPUTS:
            - mat (torch.Tensor): N1 x d Tensor to compute pairwise distances
            - mat2 (torch.Tensor): N2 x d Tensor (optional)
        
        OUTPUT:
            A similarity matrix, giving all pairwise distances for a given 
            hyperspace embedding tensor. 

            If mat2 is given, then function computes N1 x N2 tensor where
            each matrix entry M_ij is the distance from row i of mat and 
            row j of mat2. 
        """
        if mat2 is None:
            r = torch.mm(mat, mat.t())
            diag = r.diag().unsqueeze(0)
            diag = diag.expand_as(r)
            return diag + diag.t() - 2.0 * r
        else:
            r = torch.mm(mat, mat2.t())
            diag1 = torch.mm(mat, mat.t()).diag().unsqueeze(1)
            diag2 = torch.mm(mat2, mat2.t()).diag().unsqueeze(0)
            diag1 = diag1.expand_as(r)
            diag2 = diag2.expand_as(r)
            return diag1 + diag2 - 2.0 * r


    def radius_neighbor_count(self, sim):
        """
        Computes the Radius Neighbor Density Maps for a given
        embedding tensor.

        INPUTS:
            - embedding (torch.Tensor): N x d Tensor, where d is the 
            hyperspace dimension.
            - radii (list of floats): radius values for computing the
            neighbors of each point. 
            - min_points (int): minimum number of points inside a given radius
            for hinging the density loss. 

        OUTPUT:
            out (torch.Tensor): N x n_radii feature map, where each feature 
            corresponds to the normalized density value (num_points inside 
            radius r / min_points, clamped with max 1).
        """
        out = []
        for r in self._radii:
            dmap = torch.sum((sim < r), dim=1)
            out.append(dmap)
        out = torch.cat(out, dim=1)
        return out

    
    def compute_density(self, embedding, clabel):
        """
        NOTE: This function may not be differentiable. 
        """
        groups = clabel.unique(sorted=True).int()
        similarity = self.similarity_matrix(embedding)
        density_map = torch.zeros(embedding.shape[0], self._num_radii * 2)
        for i, c in enumerate(groups):
            friends = self.radius_neighbor_count(similarity[clabel == c, clabel == c]).float()
            enemies = self.radius_neighbor_count(similarity[clabel == c, clabel != c]).float()
            if self.density_mode == 0:
                density_friends = friends / (friends + enemies)
                density_enemies = enemies / (friends + enemies)
                density_loss = torch.log(density_friends + 1e-8) + torch.log(1 - density_enemies + 1e-8)
                print(density_loss)
            elif self.density_mode == 1:
                density_friends = friends / self._nsVols.expand_as(friends)
                density_enemies = enemies / self._nsVols.expand_as(enemies)
            else:
                raise ValueError("Invalid input value for density_mode")
            density = torch.cat([density_friends.view(-1, 1), density_enemies.view(-1, 1)])
            density_map[clabel == c] = density
        
        return density_map


    def forward(self, out, segment_scales, group_scales):
        print(out)
        print(segment_scales)
        print(group_scales)