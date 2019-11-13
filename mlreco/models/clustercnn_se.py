import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.loss import SpatialEmbeddingsLoss, SpatialEmbeddingsLoss2
from .cluster_cnn.spatial_embeddings import SpatialEmbeddings1


class ClusterCNN(SpatialEmbeddings1):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer. 
        - embedding_dim: dimension of final embedding space for clustering. 
    '''
    def __init__(self, cfg):
        super(ClusterCNN, self).__init__(cfg)

class ClusteringLoss(SpatialEmbeddingsLoss):
    '''
    Vanilla discriminative clustering loss applied to final embedding layer.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss, self).__init__(cfg)


class ClusteringLoss2(SpatialEmbeddingsLoss2):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss2, self).__init__(cfg)