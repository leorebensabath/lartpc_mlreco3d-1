import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from mlreco.models.layers.autoencoder import ConvAutoEncoder
from mlreco.models.layers.base import NetworkBase

class AutoEncoder(NetworkBase):

    def __init__(self, cfg, name='autoencoder'):
        super(AutoEncoder, self).__init__(cfg)

        print(cfg['modules'])

        if 'modules' in cfg:
            self.model_config = cfg['modules'][name]
        else:
            self.model_config = cfg
        print('UResNet Configs')
        print(self.model_config)

        # UResNet Configurations
        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 5)
        # Unet number of features
        self.num_filters = self.model_config.get('filters', 16)
        # UNet number of features per level
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]
        self.inputKernel = self.model_config.get('input_kernel_size', 3)

        # Input Layer Configurations and commonly used scn operations.
        self.input = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial_size, mode=3)).add(
            scn.SubmanifoldConvolution(self.dimension, self.nInputFeatures, \
            self.num_filters, self.inputKernel, self.allow_bias)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        self.add = scn.AddTable()

        # Define Sparse UResNet Encoder
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()
        for i in range(self.num_strides):
            m = scn.Sequential()
            for _ in range(self.reps):
                self._resnet_block(m, self.nPlanes[i], self.nPlanes[i])
            self.encoding_block.add(m)
            m = scn.Sequential()
            if i < self.num_strides-1:
                m.add(
                    scn.BatchNormLeakyReLU(self.nPlanes[i], leakiness=self.leakiness)).add(
                    scn.Convolution(self.dimension, self.nPlanes[i], self.nPlanes[i+1], \
                        self.downsample[0], self.downsample[1], self.allow_bias))
            self.encoding_conv.add(m)

        # Define Sparse UResNet Decoder.
        self.decoding_block = scn.Sequential()
        self.decoding_conv = scn.Sequential()
        for i in range(self.num_strides-2, -1, -1):
            m = scn.Sequential().add(
                scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                scn.FullConvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                    self.downsample[0], self.downsample[1], self.allow_bias))
            self.decoding_conv.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_autoencoder_block(m, self.nPlanes[i], self.nPlanes[i])
            self.decoding_block.add(m)

        self.sparsify = scn.Sparsify(self.dimension, 16)
        self.pred_voxels = nn.ModuleList()
        for i, f in enumerate(self.nPlanes[::-1]):
            self.pred_voxels.append(nn.Linear(f, 1))

        final_size = self.spatial_size // (2**(self.num_strides-1))

        self.global_pool = scn.Sequential(
            scn.BatchNormLeakyReLU(self.nPlanes[-1], leakiness=self.leakiness),
            scn.Convolution(self.dimension, self.nPlanes[-1], self.nPlanes[-1],
                final_size, final_size, self.allow_bias)
        )

        self.latent_dim = 256

        self.linear1 = scn.Sequential(
            scn.BatchNormLeakyReLU(self.nPlanes[-1], leakiness=self.leakiness),
            scn.NetworkInNetwork(self.nPlanes[-1], self.latent_dim, self.allow_bias)
        )

        self.linear2 = scn.Sequential(
            scn.BatchNormLeakyReLU(self.latent_dim, leakiness=self.leakiness),
            scn.NetworkInNetwork(self.latent_dim, self.nPlanes[-1], self.allow_bias)
        )

        self.global_unpool = scn.Sequential(
            scn.BatchNormLeakyReLU(self.nPlanes[-1], leakiness=self.leakiness),
            scn.FullConvolution(self.dimension, self.nPlanes[-1], 
                self.nPlanes[-1], final_size, final_size, self.allow_bias)
        )


    def encoder(self, x):
        '''
        Vanilla UResNet Encoder

        INPUTS:
            - x (scn.SparseConvNetTensor): output from inputlayer (self.input)

        RETURNS:
            - features_encoder (list of SparseConvNetTensor): list of feature
            tensors in encoding path at each spatial resolution.
        '''
        # Embeddings at each layer
        features_enc = [x]
        # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            features_enc.append(x)
            x = self.encoding_conv[i](x)

        res = {
            "features_enc": features_enc,
            "deepest_layer": x
        }

        return res


    def decoder(self, x):
        features_dec = []
        for i, layer in enumerate(self.decoding_conv):
            x = layer(x)
            x = self.decoding_block[i](x)
            features_dec.append(x)
        return features_dec



    def forward(self, input):

        point_cloud, = input
        coords = point_cloud[:, 0:self.dimension+1].float()
        features = point_cloud[:, self.dimension+1:].float()

        # Encoder
        x = self.input((coords, features))
        encoder_res = self.encoder(x)
        print(encoder_res['features_enc'])
        z = encoder_res['deepest_layer']
        z = self.global_pool(z)

        # Latent Space Linear Layer 
        latent = self.linear1(z)
        z = self.linear2(latent)

        # Decoder
        x = self.global_unpool(z)
        x = self.decoder(x)
        
        print(x)

        assert False
        return res


class AutoEncoderLoss(nn.Module):

    def __init__(self, cfg, name='ae_loss'):
        super(AutoEncoderLoss, self).__init__()
        self.l2loss = nn.MSELoss()

    def forward(self, res, input_data, cluster_label):
        reconstruction = res['reco'][0]
        target = input_data[0][:, -1].view(-1, 1).float()
        clabel = cluster_label[0][:, -2]
        batch_index = cluster_label[0][:, 3]
        loss = []
        for i, bidx in enumerate(batch_index.unique()):
            batch_mask = batch_index == bidx
            reco_batch = reconstruction[batch_mask]
            target_batch = target[batch_mask]
            clabels_batch = clabel[batch_mask]
            batch_loss = []
            for j, cidx in enumerate(clabels_batch.unique()):
                cluster_mask = clabels_batch == cidx
                reco_cluster = reco_batch[cluster_mask]
                target_cluster = target_batch[cluster_mask]
                l = self.l2loss(reco_cluster, target_cluster)
                batch_loss.append(l)
            batch_loss = sum(batch_loss) / len(batch_loss)
            loss.append(batch_loss)
        loss = sum(loss) / len(loss)
        return {'loss': loss, 'accuracy': 0}
