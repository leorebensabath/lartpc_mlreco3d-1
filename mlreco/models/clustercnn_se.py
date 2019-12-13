import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.spatial_embeddings import SpatialEmbeddings1, SpatialEmbeddings2, SpatialEmbeddings3
from .cluster_cnn.losses.spatial_embeddings import *


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


class ClusterCNN2(SpatialEmbeddings2):
    '''
    UResNet with coordinate convolution block in final layer for clustering.

    Congifurations:
        - coordConv: Option to concat coordinates to input features at
        final linear layer. 
        - embedding_dim: dimension of final embedding space for clustering. 
    '''
    def __init__(self, cfg):
        super(ClusterCNN2, self).__init__(cfg)


class ClusterCNN3(SpatialEmbeddings3):

    def __init__(self, cfg):
        super(ClusterCNN3, self).__init__(cfg)


class ClusteringLoss1(MaskBCELoss2):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss1, self).__init__(cfg)


class ClusteringLoss2(MaskBCELossBivariate):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss2, self).__init__(cfg)


class ClusteringLoss3(MaskLovaszHingeLoss):
    
    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss3, self).__init__(cfg)


class ClusteringLoss4(MaskLovaszInterLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss4, self).__init__(cfg)


class ClusteringLoss6(EllipsoidalKernelLoss):
    
    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss6, self).__init__(cfg)

class ClusteringLoss7(MaskFocalLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss7, self).__init__(cfg)

class ClusteringLoss8(MaskWeightedFocalLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss8, self).__init__(cfg)