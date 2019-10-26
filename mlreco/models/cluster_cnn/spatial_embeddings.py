import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

class SpatialEmbeddings(nn.Module):

    def __init__(self, cfg, name='spatial_embeddings'):
        super(ClusterNet, self).__init__()
        import sparseconvnet as scn
        self._model_config = cfg['modules'][name]
    
        # Data Configurations
        self._dimension = self._model_config.get('data_dim', 3)
        nInputFeatures = self._model_config.get('features', 1)
        self.spatial_size = self._model_config.get('spatial_size', 512)

        # UResNet Configurations
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        num_classes = self._model_config.get('num_classes', 5)

        # ClusterNet Configurations
        self._N = self._model_config.get('N', 0)
        self._coordConv = self._model_config.get('coordConv', False)
        self._simpleN = self._model_config.get('simple_conv', False)
        self._hypDim = self._model_config.get('hypDim', 16)

        # SeedNet Configurations
