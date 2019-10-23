import torch
import sys
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.discriminative_loss import DiscriminativeLoss
from mlreco.models.clusternet import ClusterNet


class FPN(torch.nn.Module):
    """
    Feature Pyramid Network

    Original Paper: https://arxiv.org/abs/1612.03144

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    num_classes : int
        Should be number of classes (+1 if we include ghost points directly)
    data_dim : int
        Dimension 2 or 3
    spatial_size : int
        Size of the cube containing the data, e.g. 192, 512 or 768px.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    features: int, optional
        How many features are given to the network initially
    N: int, optional
        Performs N convolution operation at each layer to transform from segmentation
        features to clustering features
    coordConv: bool, optional
        If True, network concatenates normalized coordinates (between -1 and 1) to 
        the feature tensor before the final 1x1 convolution (linear) layer. 
    simpleN: bool, optional
        Uses ResNet blocks by default. If True, N-convolution blocks are replaced with
        simple BN + SubMfdConv layers.
    hypDim: int, optional
        Dimension of clustering hyperspace.
    """

    def __init__(self, cfg, name="clusternet_fpn"):
        super(PyramidNet, self).__init__()
        self._model_config = cfg['modules'][name]

        # Model Configurations
        self._dimension = self._model_config.get('data_dim', 3)
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        self._num_strides = num_strides
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)
        self.spatial_size = self._model_config.get('spatial_size', 512)
        num_classes = self._model_config.get('num_classes', 5)
        self._N = self._model_config.get('N', 0)
        self._coordConv = self._model_config.get('coordConv', False)
        self._simpleN = self._model_config.get('simple_conv', False)
        self._hypDim = self._model_config.get('hypDim', 16)        
        self._leakiness = self._model_config.get('leakiness', 0.0)

        # Auxilary Variables
        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]  # [filter size, filter stride]

        # InputLayer
        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, self.spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, nInputFeatures, m, 3, False))

        # Encoder
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()
        for i in range(num_strides):
            module = scn.Sequential()
            for _ in range(reps):
                block(module, nPlanes[i], nPlanes[i])
            self.encoding_block.add(module)
            module2 = scn.Sequential()
            if i < num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False))
            self.encoding_conv.add(module2)

        # 1x1 Convolution Maps
        self.lateral = scn.Sequential()nPlanes_decoder
        for i in range(num_strides):
            self.lateral.add(
                scn.NetworkInNetwork(nPlanes[i], nPlanes[i+1], False)
            )
        
        # Decoder
        self.decoding_block = scn.Sequential()
        self.decoding_conv = scn.Sequential()
        for i in range(num_strides-2, -1, -1):
            module1 = scn.Sequential().add(
                scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                    downsample[0], downsample[1], False))
            self.decoding_conv.add(module1)
            module2 = scn.Sequential()
            for j in range(reps):
                block(module2, nPlanes[i], nPlanes[i])
            self.decoding_blocks.add(module2)
        
        self.elemwise_add = scn.AddTable()


    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        """
        point_cloud = input[0]
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:self._dimension+2].float()

        x = self.input((coords, features))

        encoder_layers = []
        decoder_layers = []

        res = {
            'segmentation': cnet_output['segmentation'],
            'cluster_features': [out]
            }

        return res


class ClusteringLoss(nn.Module):

    def __init__(self, cfg, reduction='sum'):
        super(ClusteringLoss, self).__init__()
        self._cfg = cfg['modules']['discriminative_loss']
        self.clustering_loss = DiscriminativeLoss(cfg)
    
    def forward(self, result, slabel, clabel):
        cluster_res = self.clustering_loss(result, slabel, clabel)
        return cluster_res