import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from mlreco.models.cluster_cnn.clustering_loss import DiscriminativeLoss, MultiScaleLoss
from mlreco.models.cluster_cnn.uresnet import ClusterUNet

class StackNet(torch.nn.Module):
    """
    StackNet

    Variant of Sparse UResNet, see documentation for details.

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

    def __init__(self, cfg, name="clusternet"):
        super(StackNet, self).__init__()
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

        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        fcsize = sum(nPlanes)
        nPlanes_decoder = [i * int(fcsize / num_strides) for i in range(num_strides, 0, -1)]
        downsample = [kernel_size, 2]  # [filter size, filter stride]

        # Clusternet Backbone (Multilayer Loss)
        self.clusternet = ClusterNet(cfg, name='clusternet')

        # UnPooling
        self.unpooling = scn.Sequential()

        for i in range(num_strides):
            if i < num_strides-1:
                module_unpool = scn.Sequential()
                for _ in range(num_strides-1-i):
                    module_unpool.add(
                        scn.UnPooling(self._dimension, downsample[0], downsample[1]))
                self.unpooling.add(module_unpool)

        # Feature Reducing Layers (NINs)
        self.cluster_decoder = scn.Sequential()
        for i in range(num_strides-1):
            module = scn.Sequential()
            module.add(
                scn.BatchNormLeakyReLU(nPlanes_decoder[i])).add(
                scn.NetworkInNetwork(nPlanes_decoder[i], nPlanes_decoder[i+1], False))
            self.cluster_decoder.add(module)

        # Output Layer for Clustering
        self.cluster_output = scn.Sequential().add(
           scn.BatchNormReLU(nPlanes_decoder[-1])).add(
           scn.OutputLayer(self._dimension))

        self.concat = scn.JoinTable()

        # Last Linear Layers (for tunable hyperspace dimension)
        self.linear = torch.nn.Linear(nPlanes_decoder[-1] + (3 if self._coordConv else 0), self._hypDim)

    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points

        Returns:
            - res (dict): dictionary of a list of tensors, where the length of each
            list corresponds to minibatch size. 

            - segmentation: semantic segmentation scores (N x 5)
            - cluster_features: clustering hyperspace embedding coordinates (N x d)
        """
        point_cloud = input[0]
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:self._dimension+2].float()

        cnet_output = self.clusternet(input)

        feature_dec = cnet_output['cluster_feature']

        # Feature Maps for Clustering
        feature_clustering = []
        for i, layer in enumerate(feature_dec[0][::-1]):
            if i < self._num_strides-1:
                f = self.unpooling[i](layer)
                feature_clustering.append(f)
            else:
                feature_clustering.append(layer)
        
        out = self.concat(feature_clustering)
        out = self.cluster_decoder(out)
        out = self.cluster_output(out)
        if self._coordConv:
            normalized_coords = (coords[:, 0:3] - self.spatial_size /2) / float(self.spatial_size)
            out = torch.cat([out, normalized_coords], dim=1)
        out = self.linear(out)

        res = {
            'segmentation': cnet_output['segmentation'],
            'cluster_features': [out]
            }

        return res