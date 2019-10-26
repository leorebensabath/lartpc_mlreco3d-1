import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

from mlreco.models.layers.uresnet import UResNet


class ClusterUNet(UResNet):

    def __init__(self, cfg, name='clusterunet'):
        '''
        Multi-Scale Clustering Network for Particle Instance Segmentation.

        ClusterUNet adds N convolution layers to each of the decoding path
        feature tensor of UResNet and outputs an embedding for each spatial
        resolution. 

        Configuration
        -------------
        N: int, optional
            Performs N convolution operation at each layer 
            to transform from segmentation features to clustering features
        coordConv: bool, optional
            If True, network concatenates normalized coordinates (between -1 and 1) to 
            the feature tensor at every spatial resolution in the decoding path.
        simpleN: bool, optional
            Uses ResNet blocks by default. If True, N-convolution blocks 
            are replaced with simple BN + SubMfdConv layers.
        embedding_dim: int, optional
            Dimension of clustering hyperspace.
        '''
        super(ClusterUNet, self).__init__(cfg, name='uresnet')
        self._num_classes = self.model_config.get('num_classes', 5)
        self._N = self.model_config.get('N', 1)
        self._simpleN = self.model_config.get('simple_conv', False)
        self._embedding_dim = self.model_config.get('embedding_dim', 8)
        self._coordConv = self.model_config.get('coordConv', False)

        if self._simpleN:
            clusterBlock = self._block
        else:
            clusterBlock = self._resnet_block

        # Intermediate transformations and clustering upsamplings
        self.cluster_conv = scn.Sequential()
        self.cluster_transform = scn.Sequential()
        for i, layer in enumerate(self.decoding_blocks):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias))
            self.cluster_conv.add(m)
            m = scn.Sequential()
            for j in range(self._N):
                clusterBlock(m, self.nPlanes[i] + \
                    (self._dimension if j == 0 else 0), self.nPlanes[i])
            self.cluster_transform.add(m)
        
        # NetworkInNetwork layer for final embedding space. 
        self.embedding = scn.NetworkInNetwork(
            self.nPlanes[-1], self._embedding_dim, self.allow_bias)


    def decoder(self, features_enc):
        '''
        ClusterUNet Decoder

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): 
            list of feature tensors in decoding path at each spatial resolution.
            - features_cluster (list of scn.SparseConvNetTensor): 
            list of transformed features on which we apply clustering loss 
            at every spatial resolution.
        '''
        features_dec = []
        features_cluster = []
        for i, layer in enumerate(self.decoding_conv):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_blocks[i](x)
            features_dec.append(x)
            emb = self.cluster_transform[i]

        result = {
            "features_dec": features_dec,
            "features_cluster": features_cluster
        }

        return result


    def forward(self, input):

        pass