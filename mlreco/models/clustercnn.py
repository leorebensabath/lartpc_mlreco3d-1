import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn


from .cluster_cnn.clusternet import ClusterUNet
from .cluster_cnn.cluster_loss import MultiScaleLoss, AllyEnemyLoss

class ClusterCNN(ClusterUNet):

    def __init__(self, cfg):
        super(ClusterCNN, self).__init__(cfg)


class ClusteringLoss(nn.Module):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss, self).__init__()
        self.model_config = cfg['modules'][name]

        # Enhancement Configurations
        multiscale_loss = self.model_config.get('multiscale_loss', True)
        affinity_loss = self.model_config.get('affinity_loss', True)
        attention_weighting = self.model_config.get('attention_weighting', False)

        self.loss_func = AllyEnemyLoss(cfg)

    def forward(self, out, segment_label, cluster_label):

        result = self.loss_func(out, segment_label, cluster_label)
        return result

