import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from .cluster_cnn.utils import add_normalized_coordinates
from .cluster_cnn.loss import DiscriminativeLoss
from mlreco.models.uresnet import UResNet

class ClusterUNet(UResNet):
    '''
    Clustering model with UResNet Backbone, where we optimize
    the network with a discriminative loss function for clustering.

    Loss is applied only at the final layer that represents the
    learned embedding space.
    '''
    def __init__(self, cfg, name='clusterunet_single'):
        super(ClusterUNet, self).__init__(cfg, name='uresnet')
        self._coordConv = self.model_config.get('coordConv', False)
        self._embedding_dim = self.model_config.get('embedding_dim', 8)
        if self._coordConv:
            self.linear = torch.nn.Linear(m + self._dimension, self._embedding_dim)
        else:
            self.linear = torch.nn.Linear(m, self._embedding_dim)


    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        point_cloud, = input
        coords = point_cloud[:, :-2].float()
        features = point_cloud[:, -1][:, None].float()
        fout = self.sparseModel((coords, features))

        if self._coordConv:
            normalized_coords = (coords - self.spatial_size / 2)\
                / float(self.spatial_size / 2)
            fout = torch.cat([normalized_coords, fout], dim=1)
        else:
            fout = self.sparseModel((coords, features))
        embedding = self.linear(fout)

        return {
            'cluster_feature': [embedding]
        }


class ClusteringLoss(DiscriminativeLoss):

    def __init__(self, cfg, name='cluster_single_loss'):
        super(ClusteringLoss, self).__init__(cfg)

    