import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.adaptis import *


class ClusterCNN(AdaptIS):

    def __init__(self, cfg, name='adaptis'):
        super(ClusterCNN, self).__init__(cfg, name=name)


class ClusteringLoss(nn.Module):

    def __init__(self, cfg, name='clustering_loss'):
        super(ClusteringLoss, self).__init__()
        self._model_config = cfg['modules'][name]
        self._dloss = DiscriminativeLoss(cfg, name=name)


    def forward(self, out, semantic_labels, group_labels, particle_labels):

        segmentation = out['segmentation'][0]
        attention = out['attention_map'][0]
        cluster_masks = out['cluster_masks'][0]

        return self._dloss(out, semantic_labels, group_labels)