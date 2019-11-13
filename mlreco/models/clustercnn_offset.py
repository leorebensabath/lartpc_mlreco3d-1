import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.loss import FixedCentroidLoss
from mlreco.models.discriminative_loss import UResNet


class ClusterCNN(UResNet):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer. 
        - embedding_dim: dimension of final embedding space for clustering. 
    '''
    def __init__(self, cfg):
        super(ClusterCNN, self).__init__(cfg)

class ClusteringLoss(FixedCentroidLoss):
    '''
    Vanilla discriminative clustering loss applied to final embedding layer.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss, self).__init__(cfg)