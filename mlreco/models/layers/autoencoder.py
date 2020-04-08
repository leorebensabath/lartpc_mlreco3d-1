import torch
import torch.nn as nn
import sparseconvnet as scn
import pprint

from mlreco.models.layers.base import NetworkBase

class ConvAutoEncoder(NetworkBase):

    def __init__(self, cfg, name='conv_autoenc'):
        super(ConvAutoEncoder, self).__init__(cfg)
        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg
        pprint.pprint(self.model_config)

        # Network Configurations
        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 6)
        # Unet number of features
        self.num_filters = self.model_config.get('filters', 16)
        # UNet number of features per level
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]
        self.inputKernel = self.model_config.get('input_kernel_size', 3)
        self.final_size = self.spatial_size / (2**(self.num_strides - 1))
        self.latent_dim = 16

        # Input Layer Configurations and commonly used scn operations.
        self.input = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial_size, mode=3)).add(
            scn.SubmanifoldConvolution(self.dimension, self.nInputFeatures, \
            self.num_filters, self.inputKernel, self.allow_bias)) # Kernel size 3, no bias

        # Define Sparse UResNet Encoder
        self.encoding_blocks = scn.Sequential()
        for i in range(self.num_strides):
            m = scn.Sequential()
            for _ in range(self.reps):
                self._resnet_block(m, self.nPlanes[i], self.nPlanes[i])
            self.encoding_blocks.add(m)
            m = scn.Sequential()
            if i < self.num_strides-1:
                m.add(
                    scn.BatchNormLeakyReLU(self.nPlanes[i], leakiness=self.leakiness)).add(
                    scn.Convolution(self.dimension, self.nPlanes[i], self.nPlanes[i+1], \
                        self.downsample[0], self.downsample[1], self.allow_bias))
            self.encoding_blocks.add(m)

        self.decoding_blocks = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.TransposeConvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias)).add(
                scn.SparsifyFCS(self.dimension))
            self.decoding_blocks.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i], self.nPlanes[i])
            self.decoding_blocks.add(m)

        self.pool = scn.Convolution(self.dimension, self.nPlanes[-1], self.nPlanes[-1],
            self.final_size, self.final_size, self.allow_bias)
        self.unpool = scn.Sequential().add(
            scn.TransposeConvolution(self.dimension, self.nPlanes[-1], self.nPlanes[-1],
                    self.final_size, self.final_size, self.allow_bias)).add(
            scn.SparsifyFCS(self.dimension))
        self.output = scn.Sequential().add(
            scn.NetworkInNetwork(self.nPlanes[0], 1, self.allow_bias)).add(
            scn.OutputLayer(self.dimension))

        self.linear1 = scn.NetworkInNetwork(self.nPlanes[-1], self.latent_dim, self.allow_bias)
        self.linear2 = scn.NetworkInNetwork(self.latent_dim, self.nPlanes[-1], self.allow_bias)

    def encoder(self, x):
        for i, layer in enumerate(self.encoding_blocks):
            x = layer(x)
        x = self.pool(x)
        return x

    def fc1(self, x):
        return self.linear1(x)

    def fc2(self, z):
        return self.linear2(z)

    def decoder(self, x):
        x = self.unpool(x)
        for i, layer in enumerate(self.decoding_blocks):
            x = layer(x)
        return x

    def forward(self, input):
        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()

        x = self.input((coords, features))
        x = self.encoder(x)
        z = self.fc1(x)
        x = self.fc2(z)
        x = self.decoder(x)
        x = self.output(x)
        return {'reco': [x]}
