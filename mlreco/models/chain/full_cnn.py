import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict
import pprint

from mlreco.models.layers.uresnet import UResNet
from mlreco.models.layers.stacknet import StackUNet
from mlreco.models.cluster_cnn.spatial_embeddings import SpatialEmbeddings1

class FullCNN(SpatialEmbeddings1):


    def __init__(self, cfg, name='spatial_embeddings'):
        super(FullCNN, self).__init__(cfg, name=name)
        self.embedding_dim = self.model_config.get('embedding_dim', 4)
        self.coordConv = self.model_config.get('coordConv', False)
        self.num_classes = self.model_config.get('num_classes', 5)

        # Add segmentation decoder
        self.decoding_block3 = scn.Sequential()
        self.decoding_conv3 = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias))
            self.decoding_conv3.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i] * (2 if j == 0 else 1), self.nPlanes[i])
            self.decoding_block3.add(m)

        self.outputSegmentation = scn.Sequential()
        self._nin_block(self.outputSegmentation, self.num_filters, self.num_classes)


    def seg_decoder(self, features_enc, deepest_layer):
        '''
        Decoder for seediness map.

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution.
        '''
        features_seg = []
        x = deepest_layer
        for i, layer in enumerate(self.decoding_conv3):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_block3[i](x)
            features_seg.append(x)
        return features_seg


    def forward(self, input):
        '''
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points

        RETURNS:
            - feature_enc: encoder features at each spatial resolution.
            - feature_dec: decoder features at each spatial resolution.
        '''
        point_cloud, = input
        # print("Point Cloud: ", point_cloud)
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()
        features = features[:, -1].view(-1, 1)

        normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_cluster = self.decoder(features_enc, deepest_layer)
        features_seediness = self.seed_decoder(features_enc, deepest_layer)
        features_segmentation = self.seg_decoder(features_enc, deepest_layer)

        segmentation = self.outputSegmentation(features_segmentation[-1])
        embeddings = self.outputEmbeddings(features_cluster[-1])
        embeddings[:, :self.dimension] = self.tanh(embeddings[:, :self.dimension])
        embeddings[:, :self.dimension] += normalized_coords
        sigma = 2 * self.sigmoid(embeddings[:, self.dimension:self.dimension+3])
        # embeddings[:, self.dimension:self.dimension+3] = \
        #     self.softplus(embeddings[:, self.dimension:self.dimension+3])
        # embeddings[:, self.dimension+3:] = \
        #     self.tanhshrink(embeddings[:, self.dimension+3:])
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            "embeddings": [embeddings[:, :self.dimension]],
            "margins": [margins],
            "seediness": [self.sigmoid(seediness)],
            "segmentation": [segmentation]
        }

        return res
