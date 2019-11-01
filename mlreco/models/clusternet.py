import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn import cluster_model_construct, backbone_construct
from mlreco.models.layers.uresnet import UResNet
from mlreco.models.layers.fpn import FPN
from mlreco.models.layers.base import NetworkBase

###########################################################
#
# Define one multilayer model to incorporate all options.
#
# Embedding Transforming Convolutions are added on top of 
# backbone decoder features. 
# 
# Distance Estimation Map is added on top of final layer of
# backbone decoder concatenated with final layer of clustering. 
#
###########################################################

class ClusterCNN(NetworkBase):
    '''
    CNN Based Multiscale Clustering Module.

    Configurations
    ----------------------------------------------------------
    backbone: the type of backbone architecture to be used in the
    feature extraction stages. Currently the following optinos are
    available:
        - UResNet: Vanilla UResNet
        - FPN: Feature Pyramid Networks

    clustering: configurations for clustering transformation convolutions
    at each decoding path.

    proximity: configurations for final distance estimation map.
    ---------------------------------------------------------- 
    '''
    def __init__(self, cfg, name='clusternet'):
        super(ClusterCNN, self).__init__(cfg)

        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg

        self.backbone_config = self.model_config.get('backbone', None)
        self.clustering_config = self.model_config.get('clustering', None)
        self.proximity_config = self.model_config.get('proximity', None)

        # Construct Backbone
        net = backbone_construct(self.backbone_config.get('name', 'uresnet'))
        self.net = net(self.backbone_config)
        self.num_filters = self.net.num_filters

        # Add N-Convolutions for Clustering
        if self.clustering_config is not None:
            clusternet = cluster_model_construct(
                self.clustering_config.get('name', 'clusternet'))
            self.net = clusternet(self.clustering_config, self.net)
        
        # Add Distance Estimation Layer
        if self.proximity_config is not None:
            self.dist_N = self.proximity_config.get('dist_N', 3)
            self.dist_simple_conv = self.proximity_config.get('dist_simple_conv', False)
            self.distance_estimate = scn.Sequential()
            if self.dist_simple_conv:
                distanceBlock = self._block
            else:
                distanceBlock = self._resnet_block
            for i in range(self.dist_N):
                if i == self.dist_N-1:
                    num_output = 2
                else:
                    num_output = self.num_filters
                distanceBlock(self.distance_estimate, self.num_filters, num_output)


    def forward(self, input):
        '''

        '''

        pass


class ClusteringLoss(nn.Module):
    '''
    Loss Function for CNN Based Cascading Clustering.

    Configurations
    ----------------------------------------------------------
    loss_segmentation: 
        - configurations for semantic segmentation loss. 

    loss_clustering:
        - configurations for clustering loss.

    loss_distance:
        - configurations for distance estimation loss.
    ----------------------------------------------------------
    '''
    pass