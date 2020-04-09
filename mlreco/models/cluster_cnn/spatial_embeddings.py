import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict
import pprint

from mlreco.models.layers.uresnet import UResNet
from mlreco.models.layers.stacknet import StackUNet
from .utils import add_normalized_coordinates, distance_matrix

class SpatialEmbeddings1(UResNet):

    def __init__(self, cfg, name='spatial_embeddings'):
        super(SpatialEmbeddings1, self).__init__(cfg, name='uresnet')
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg
        print("SpatialEmbeddings1")
        pprint.pprint(self.model_config)
        self.seedDim = self.model_config.get('seediness_dim', 1)
        self.embedding_dim = self.model_config.get('embedding_dim', 3)
        self.sigmaDim = self.model_config.get('sigma_dim', 1)
        self.seed_freeze = self.model_config.get('seed_freeze', False)
        self.coordConv = self.model_config.get('coordConv', False)
        # Define Separate Sparse UResNet Decoder for seediness.
        self.decoding_block2 = scn.Sequential()
        self.decoding_conv2 = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias))
            self.decoding_conv2.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i] * (2 if j == 0 else 1), self.nPlanes[i])
            self.decoding_block2.add(m)

        # Define outputlayers
        self.outputEmbeddings = scn.Sequential()
        self._nin_block(self.outputEmbeddings, self.num_filters, self.embedding_dim + self.sigmaDim)
        self.outputEmbeddings.add(scn.OutputLayer(self.dimension))
        self.outputSeediness = scn.Sequential()
        self._nin_block(self.outputSeediness, self.num_filters, self.seedDim)
        self.outputSeediness.add(scn.OutputLayer(self.dimension))

        if self.seed_freeze:
            print('Seediness Branch Freezed')
            for p in self.decoding_block2.parameters():
                p.requires_grad = False
            for p in self.decoding_conv2.parameters():
                p.requires_grad = False
            for p in self.outputSeediness.parameters():
                p.requires_grad = False

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def seed_decoder(self, features_enc, deepest_layer):
        '''
        Decoder for seediness map.

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution.
        '''
        features_seediness = []
        x = deepest_layer
        for i, layer in enumerate(self.decoding_conv2):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_block2[i](x)
            features_seediness.append(x)
        return features_seediness


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
        # point_cloud, = input
        # coords = point_cloud[:, 0:self.dimension+1].float()
        # features = point_cloud[:, self.dimension+1:].float()
        # features = features[:, -1].view(-1, 1)
        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1]
        normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        normalized_coords = normalized_coords.float()
        features = point_cloud[:, self.dimension+1:].float()
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_cluster = self.decoder(features_enc, deepest_layer)
        features_seediness = self.seed_decoder(features_enc, deepest_layer)

        # normalized_coords = (coords[:, :3] - self.spatial_size / 2) \
        #     / (self.spatial_size / 2)

        embeddings = self.outputEmbeddings(features_cluster[-1])
        embeddings[:, :self.dimension] = self.tanh(embeddings[:, :self.dimension])
        embeddings[:, :self.dimension] += normalized_coords
        sigma = 2 * self.sigmoid(embeddings[:, self.dimension:self.dimension+3])
        l = embeddings[:, self.dimension+3:]
        margins = torch.cat([sigma, l], dim=1)
        # embeddings[:, self.dimension:self.dimension+3] = \
        #     self.softplus(embeddings[:, self.dimension:self.dimension+3])
        # embeddings[:, self.dimension+3:] = \
        #     self.tanhshrink(embeddings[:, self.dimension+3:])
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            "embeddings": [embeddings[:, :self.dimension]],
            "margins": [margins],
            "seediness": [self.sigmoid(seediness)],
            "features_cluster": [features_cluster]
        }
        # print(res)

        return res


class SpatialEmbeddings2(StackUNet):

    def __init__(self, cfg, name='spatial_embeddings'):
        super(SpatialEmbeddings2, self).__init__(cfg, name=name)
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg
        self.seedDim = self.model_config.get('seediness_dim', 1)
        self.sigmaDim = self.model_config.get('sigma_dim', 1)
        # Define Separate Sparse UResNet Decoder for seediness.
        self.decoding_block2 = scn.Sequential()
        self.decoding_conv2 = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias))
            self.decoding_conv2.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i] * (2 if j == 0 else 1), self.nPlanes[i])
            self.decoding_block2.add(m)

        # Define outputlayers
        self.embedding = None

        self.outputEmbeddings = scn.Sequential()
        self._nin_block(self.outputEmbeddings, self.num_filters, self.dimension + self.sigmaDim)
        self.outputEmbeddings.add(scn.OutputLayer(self.dimension))
        self.outputSeediness = scn.Sequential()
        self._nin_block(self.outputSeediness, self.num_filters, self.seedDim)
        self.outputSeediness.add(scn.OutputLayer(self.dimension))

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def seed_decoder(self, features_enc, deepest_layer):
        '''
        Decoder for seediness map.

        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.

        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution.
        '''
        features_seediness = []
        x = deepest_layer
        for i, layer in enumerate(self.decoding_conv2):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_block2[i](x)
            features_seediness.append(x)
        return features_seediness


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
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_dec = self.decoder(features_enc, deepest_layer)
        features_dec = [deepest_layer] + features_dec
        stack_feature = []

        for i, layer in enumerate(features_dec):
            if i < self.num_strides-1:
                f = self.unpooling[i](layer)
                stack_feature.append(f)
            else:
                stack_feature.append(layer)

        stack_feature = self.concat(stack_feature)
        out = self.cluster_decoder(stack_feature)
        features_seediness = self.seed_decoder(features_enc, deepest_layer)

        embeddings = self.outputEmbeddings(out)
        embeddings[:, :self.dimension] = self.tanh(embeddings[:, :self.dimension])
        embeddings[:, :self.dimension] += coords[:, :self.dimension] / self.spatial_size
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            "embeddings": [embeddings[:, :self.dimension]],
            "margins": [self.sigmoid(embeddings[:, self.dimension:])],
            "seediness": [self.sigmoid(seediness)]
        }

        return res


class SpatialEmbeddings3(SpatialEmbeddings1):


    def __init__(self, cfg, name='spatial_embeddings'):
        super(SpatialEmbeddings3, self).__init__(cfg, name=name)
        self.coordConv = self.model_config.get('coordConv', True)

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
        coords = point_cloud[:, 0:self.dimension+1]
        coords_temp = coords.detach().cpu().numpy()
        perm = np.lexsort((coords_temp[:, 2], coords_temp[:, 1],
                           coords_temp[:, 0], coords_temp[:, 3]))
        coords = coords[perm].float()
        normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        features = point_cloud[:, self.dimension+1:].float()
        features = features[perm]
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_cluster = self.decoder(features_enc, deepest_layer)
        features_seediness = self.seed_decoder(features_enc, deepest_layer)

        normalized_coords = (coords[:, :3] - self.spatial_size / 2) \
            / (self.spatial_size / 2)
        embeddings = self.outputEmbeddings(features_cluster[-1])
        embeddings[:, :self.embedding_dim] = embeddings[:, :self.embedding_dim]
        margins = 2 * self.sigmoid(
            embeddings[:, self.embedding_dim:self.embedding_dim+self.sigmaDim])
        # embeddings[:, self.dimension:self.dimension+3] = \
        #     self.softplus(embeddings[:, self.dimension:self.dimension+3])
        # embeddings[:, self.dimension+3:] = \
        #     self.tanhshrink(embeddings[:, self.dimension+3:])
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            "embeddings": [embeddings[:, :self.dimension]],
            "margins": [margins],
            "seediness": [self.sigmoid(seediness)],
            "features_cluster": [features_cluster],
            "coords": [coords]
        }
        # print(res)

        return res


class SpatialEmbeddings4(SpatialEmbeddings3):


    def __init__(self, cfg, name='spatial_embeddings'):
        super(SpatialEmbeddings4, self).__init__(cfg, name=name)
        self.coordConv = self.model_config.get('coordConv', True)

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
        coords = point_cloud[:, 0:self.dimension+1]
        coords_temp = coords.detach().cpu().numpy()
        perm = np.lexsort((coords_temp[:, 2], coords_temp[:, 1],
                           coords_temp[:, 0], coords_temp[:, 3]))
        coords = coords[perm].float()
        normalized_coords = (coords[:, :3] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        features = point_cloud[:, self.dimension+1:].float()
        features = features[perm]
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)

        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']
        features_cluster = self.decoder(features_enc, deepest_layer)
        features_seediness = self.seed_decoder(features_enc, deepest_layer)

        normalized_coords = (coords[:, :3] - self.spatial_size / 2) \
            / (self.spatial_size / 2)
        embeddings = self.outputEmbeddings(features_cluster[-1])
        embeddings_normalized = embeddings[:, :self.embedding_dim] \
            / torch.norm(embeddings[:, :self.embedding_dim], dim=1, keepdim=True)
        margins = 2 * self.sigmoid(
            embeddings[:, self.embedding_dim:self.embedding_dim+self.sigmaDim])
        # embeddings[:, self.dimension:self.dimension+3] = \
        #     self.softplus(embeddings[:, self.dimension:self.dimension+3])
        # embeddings[:, self.dimension+3:] = \
        #     self.tanhshrink(embeddings[:, self.dimension+3:])
        seediness = self.outputSeediness(features_seediness[-1])

        res = {
            "embeddings": [embeddings_normalized],
            "margins": [margins],
            "seediness": [self.sigmoid(seediness)],
            "features_cluster": [features_cluster],
            "coords": [coords]
        }
        # print(res)

        return res
