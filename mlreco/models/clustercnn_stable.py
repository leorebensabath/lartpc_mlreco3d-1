import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.clusternet import ClusterUNet
from .cluster_cnn.cluster_loss import EnhancedEmbeddingLoss, MultiScaleLoss

class ClusterCNN(ClusterUNet):
    '''
    UResNet with multi-scale convolution blocks for clustering at
    each spatial resolution.
    '''
    def __init__(self, cfg):
        super(ClusterCNN, self).__init__(cfg)


class ClusteringLoss(nn.Module):
    '''
    
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss, self).__init__()
        self.model_config = cfg['modules'][name]

        # TODO: Define single model with configurable enhancements. 

        self.loss_func = EnhancedEmbeddingLoss(cfg)

    def forward(self, out, segment_label, cluster_label):

        result = self.loss_func(out, segment_label, cluster_label)
        return result