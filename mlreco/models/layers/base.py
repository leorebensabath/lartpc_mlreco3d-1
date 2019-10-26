import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict


class NetworkBase(nn.Module):
    '''
    Base Class for UResNet-type architectures.

    Configurations
    -------------
    data_dim : int
        Dimension 2 or 3
    spatial_size : int
        Size of the cube containing the data, e.g. 192, 512 or 768px.
    allow_bias: bool, optional
        Option to allow bias term in convolution layers.
    features: int, optional
        How many features are given to the network initially
    leakiness: float, optional
        slope value for LeakyReLU activation functions.
    '''
    def __init__(self, cfg, name='network_base'):
        super(NetworkBase, self).__init__()
        self._model_config = cfg['modules'][name]
        # Cross-network module configurations
        self._dimension = self.model_config.get('data_dim', 3)
        self._nInputFeatures = self.model_config.get('features', 1)
        self._spatial_size = self.model_config.get('spatial_size', 512)
        self._leakiness = self.model_config.get('leakiness', 0.0)
        self._allow_bias = self.model_config.get('allow_bias', False)

    # Include cross-network utility functions and submodules here.

    def _resnet_block(self, module, a, b):
        '''
        Utility Method for attaching ResNet-Style Blocks.

        INPUTS:
            - module (scn Module): network module to attach ResNet block.
            - a (int): number of input feature dimension
            - b (int): number of output feature dimension

        RETURNS:
            None (operation is in-place)
        '''
        module.add(scn.ConcatTable()
            .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, self._allow_bias))
            .add(scn.Sequential()
                .add(scn.BatchNormLeakyReLU(a, leakiness=self._leakiness))
                .add(scn.SubmanifoldConvolution(self._dimension, a, b, 3, self._allow_bias))
                .add(scn.BatchNormLeakyReLU(b, leakiness=self._leakiness))
                .add(scn.SubmanifoldConvolution(self._dimension, b, b, 3, self._allow_bias)))
        ).add(scn.AddTable())

    
    def _block(self, module, a, b):
        '''
        Utility Method for attaching 2 x (Conv-BN) Blocks.

        INPUTS:
            - module (scn Module): network module to attach ResNet block.
            - a (int): number of input feature dimension
            - b (int): number of output feature dimension

        RETURNS:
            None (operation is in-place)
        '''
        module.add(scn.Sequential()
            .add(scn.BatchNormLeakyReLU(a, leakiness=self._leakiness))
            .add(scn.SubmanifoldConvolution(self._dimension, a, b, 3, self._allow_bias))
            .add(scn.BatchNormLeakyReLU(b, leakiness=self._leakiness))
            .add(scn.SubmanifoldConvolution(self._dimension, b, b, 3, self._allow_bias))
        )


    def forward(self, input):
        raise NotImplementedError