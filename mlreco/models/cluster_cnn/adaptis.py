import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn
from collections import defaultdict

# Pytorch Implementation of AdaptIS
# Original Paper: https://arxiv.org/pdf/1909.07829.pdf

from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

from mlreco.models.uresnet_lonely import UResNet
from mlreco.models.discriminative_loss import DiscriminativeLoss


class AdaIN(nn.Module):
    '''
    Adaptive Instance Normalization Layer
    Original Paper: https://arxiv.org/pdf/1703.06868.pdf

    Many parts of the code is borrowed from pytorch original
    BatchNorm implementation. 

    INPUT:
        - input: SparseTensor

    RETURNS:
        - out: SparseTensor
    '''
    __constants__ = ['momentum', 'eps', 'weight', 'bias',
                     'num_features', 'affine']

    def __init__(self, num_features, eps=1e-5):
        super(AdaIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.reset_parameters()
        assert (weight.shape[1] == bias.shape[1] == num_features)
        self._weight = 1.0
        self._bias = 0.0
    
    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, weight):
        '''
        Set weight and bias parameters for AdaIN Layer. 
        Note that in AdaptIS, the parameters to the AdaIN layer
        are trainable outputs from the controller network. 
        '''
        if weight.shape[1] != num_features:
            raise ValueError('Supplied weight vector feature dimension\
             does not match layer definition!')
        self._weight = weight
    
    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        if bias.shape[1] != num_features:
            raise ValueError('Supplied bias vector feature dimension\
             does not match layer definition!')
        self._bias = bias

    def forward(self, input):
        '''
        INPUTS:
            - input (N x d SCN SparseTensor)
        RETURNS:
            - out (N x d SCN SparseTensor)
        '''
        out = scn.SparseConvNetTensor()
        out.metadata = input.metadata
        out.spatial_size = input.spatial_size
        means = torch.mean(input.features, dim=1)
        variances = torch.pow(torch.std(input, dim=1), 2)
        out.features = (input - means) / torch.sqrt(variances + eps) * self.weight + self.bias
        return out


class ControllerNet(nn.Module):
    '''
    MLP Network Producing AdaIN Parameter Vector.
    '''
    def __init__(self, num_input, num_output, depth=3, leakiness=0.0,
                 hidden_dims=None):
        '''
        Simple MLP Module to Transfer from Encoder extracted features to
        AdaIN input parameters.
        '''
        super(ControllerNet, self).__init__()
        self.leakiness = leakiness
        modules = []
        if hidden_dims is not None:
            assert (len(hidden_dims) == depth-1)
            dims = [num_input] + hidden_dims + [num_output]
        else:
            dims = [num_input] + [num_output] * depth
        for i in range(depth):
            modules.append(nn.BatchNorm1d(dims[i]))
            modules.append(nn.LeakyReLU(negative_slope=self.leakiness))
            modules.append(nn.Linear(dims[i], dims[i+1]))
        self.net = nn.Sequential(*modules)

    def forward(self, input):

        return self.net(input)
        

class RelativeCoordConv(nn.Module):
    '''
    Relative Coordinate Convolution Blocks introduced in AdaptIS paper.
    We tailer to our use (Sparse Tensors). 

    This serves as a prior on the location of the object.

    Original paper contains an "instance size" limiting parameter R,
    which may not suit our purposes. 
    '''
    def __init__(self, num_input, num_output, data_dim=3, spatial_size=512, leakiness=0.0):
        super(RelativeCoordConv, self).__init__()
        self._num_input = num_input
        self._num_output = num_output
        self._data_dim = data_dim
        self._spatial_size = spatial_size

        # CoordConv Block Definition
        self.net = scn.Sequential()
        self.net.add(
            scn.SubmanifoldConvolution(data_dim, num_input + data_dim, num_output, 3, False)).add(
                scn.BatchNormLeakyReLU(num_output, leakiness=leakiness))


    def forward(self, input, point):
        '''
        INPUTS:
            - input (N x num_input)
            - coords (N x data_dim)
            - point (1 x data_dim)
        '''
        coords = input.get_spatial_locations()
        out = scn.SparseConvNetTensor()
        out.metadata = input.metadata
        out.spatial_size = input.spatial_size
        normalized_coords = (coords - point) / float(self._spatial_size)
        out.features = torch.cat([normalized_coords, input.features], dim=1)
        out = self.net(out)

        return out


class AdaptIS(nn.Module):
    '''
    Wrapper module for entire AdaptIS network chain.

    We roughly follow the network architecture description 
    in page 6 of paper: https://arxiv.org/pdf/1909.07829.pdf.

    We rename "point proposal branch" in the paper as "attention proposal",
    to avoid confusion with existing PPN. 
    '''

    def __init__(self, cfg, name='AdaptIS'):
        super(AdaptIS, self).__init__()
        self._model_config = cfg['modules'][name]

        # Model Configurations
        self._leakiness = self._model_config.get('leakiness', 0.0)
        self.feature_size = self._model_config.get('feature_size', 32)
        self.attention_depth = self._model_config.get('attention_depth', 3)
        self.segmentation_depth = self._model_config.get('segmentation_depth', 3)
        self.attention_hidden = self._model_config.get('attention_hidden', 32)
        self.segmentation_hidden = self._model_config.get('segmentation_hidden', 32)
        self._N = self._model_config.get('N', 3)
        self.instance_decoder_depth = self._model_config.get('instance_decoder_depth', 3)
        self.train = self._model_config.get('train', True)

        # TODO: Give option to use ResNet Blocks insteaed of Conv+BN+LeakyReLU Blocks

        # Backbone Feature Extraction Network
        self.net = UResNet(cfg, name='uresnet_lonely')
        num_classes = self._model_config.get('num_classes', 5)

        # Attention Proposal Branch
        self.attention_net = scn.Sequential()
        for i in range(self.attention_depth):
            module = scn.Sequential()
            module.add(
                scn.SubmanifoldConvolution(self.net._dimension,
                (self.feature_size if i == 0 else self.attention_hidden),
                self.attention_hidden, 3, False)).add(
                scn.BatchNormLeakyReLU(self.attention_hidden, leakiness=self._leakiness))
            self.attention_net.add(module)
        self.attention_net.add(scn.NetworkInNetwork(self.attention_hidden, 1, False))

        # Segmentation Branch
        self.segmentation_net = scn.Sequential()
        for i in range(self.segmentation_depth):
            module = scn.Sequential()
            module.add(
                scn.SubmanifoldConvolution(self.net._dimension,
                (self.feature_size if i == 0 else self.segmentation_hidden),
                self.segmentation_hidden, 3, False)).add(
                scn.BatchNormLeakyReLU(self.segmentation_hidden, leakiness=self._leakiness))
            self.segmentation_net.add(module)
        self.segmentation_net.add(scn.NetworkInNetwork(self.attention_hidden, num_classes, False))

        # Instance Selection Branch
        # 1. Controller Network makes AdaIN parameter vector from query point. 
        self.controller_weight = ControllerNet(feature_size, feature_size, 3)
        self.controller_bias = ControllerNet(feature_size, feature_size, 3)
        # 2. Relative CoordConv and concat to feature tensor
        self.rel_cc = RelativeCoordConv(feature_size, feature_size)
        self.concat = scn.JoinTable()
        # 3. Several Convolution Layers
        self.instance_net = scn.Sequential()
        for i in range(N):
            module = scn.Sequential()
            module.add(
                scn.SubmanifoldConvolution(self.net._dimension,
                    feature_size, feature_size, 3, False)).add(
                scn.BatchNormLeakyReLU(feature_size, leakiness=self._leakiness))
            self.instance_net.add(module)
        # 4. Adaptive Instance Normalization 
        self.adapt_in = AdaIN(feature_size)
        # 5. Mask Generating Decoder
        instance_downsample = [feature_size] + [int(feature_size / 2**i) for i in range(self.instance_decoder_depth)]
        self.instance_dec = scn.Sequential()
        for i in range(self.instance_decoder_depth):
            module = scn.Sequential()
            module.add(
                scn.SubmanifoldConvolution(self.net._dimension,
                    instance_downsample[i], instance_downsample[i+1], 3, False)).add(
                scn.BatchNormLeakyReLU(instance_downsample[i+1], leakiness=self._leakiness))
            self.instance_dec.add(module)
        # Final Mask is Binary
        self.instance_dec.add(scn.NetworkInNetwork(instance_downsample[-1], 1, False))


    def find_query_points(self, attention_map):
        '''
        TODO:
        Based on attention map, find query points to be passed to 
        AdaIN layers via local maximum finding.
        '''
        pass

        
    def forward(self, input, query_points):
        '''
        INPUTS:
            - input: usual input to UResNet
            - query_points: list of tensor indices (full spatial resolution) of
            sampled query points for mask generation. Only used during training.

            During training, we sample random points at least once from each cluster.
            During inference, we sample the highest attention scoring points.
        
        TODO: Turn off attention map training. 
        '''

        net_output = self.net(input)
        # Get last feature layer in UResNet
        feature_tensor = net_output['ppn_feature_dec'][0][-1].features
        coords = feature_tensor.get_spatial_locations()
        batch_idx = coords[:, -1].int().unique()

        # Attention map and Segmentation Map is Holistic. 
        attention_map = self.attention_net(feature_tensor)
        segmentation = self.segmentation_net(feature_tensor)

        # For Instance Branch, mask generation is instance-by-instance.
        instance_masks = []

        for i, bidx in enumerate(batch_idx):
            batch_mask = coords[:, -1] == bidx
            coords_batch = coords[batch_mask]
            feature_batch = feature_tensor[batch_mask]
            if self.train:
                # Sample K attention proposals, K = num_clusters
                pts = query_points[bidx]
                sampled_features = feature_batch[pts]
                for sample in sampled_features:
                    weight = self.controller_weight(sample)
                    bias = self.controller_bias(sample)
                    self.adapt_in.set_params(weight, bias)
                    x = self.instance_net(feature_batch)
                    x = self.adapt_in(x)
                    x = self.instance_dec(x) # Mask features, later apply BCE.
                    instance_masks.append(x)
            else:
                pass
                # TODO: Sample attention proposals by local max in attention map. 
        res = {
            'segmentation': [segmentation],
            'attention_map': [attention_map],
            'cluster_masks': [instance_masks]
        }

        return res


class ClusteringLoss(nn.Module):

    def __init__(self, cfg, name='discriminative_loss'):
        super(ClusteringLoss, self).__init__()
        self._model_config = cfg['modules'][name]
        self._dloss = DiscriminativeLoss(cfg)


    def forward(self, out, semantic_labels, group_labels):

        segmentation = out['segmentation'][0]
        attention = out['attention_map'][0]
        cluster_masks = out['cluster_masks'][0]

        return self._dloss(out, semantic_labels, group_labels)
