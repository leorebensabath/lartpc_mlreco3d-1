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

from mlreco.models.layers.uresnet import UResNet
from mlreco.models.discriminative_loss import DiscriminativeLoss
from mlreco.models.layers.base import NetworkBase
from scipy.spatial import cKDTree


class MaskSparseTensor(nn.Module):
    '''
    Module for masking Sparse Tensors
    '''
    def __init__(self, dimension=3):
        self.dimension = dimension
        self._mask = None
    
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    def forward(self, x):
        coords = x.get_spatial_locations()
        features = x.features
        out = scn.InputBatch(dimension, x.spatial_size)
        out.set_locations(coords[self._mask], torch.Tensor(features.data.shape))
        out.features = features[self._mask]
        return out


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


class AdaptIS(NetworkBase):
    '''
    Wrapper module for entire AdaptIS network chain.

    We roughly follow the network architecture description 
    in page 6 of paper: https://arxiv.org/pdf/1909.07829.pdf.

    We rename "point proposal branch" in the paper as "attention proposal",
    to avoid confusion with existing PPN. 
    '''

    def __init__(self, cfg, name='AdaptIS'):
        super(AdaptIS, self).__init__()
        self.model_config = cfg['modules'][name]

        # Model Configurations
        self.feature_size = self._model_config.get('feature_size', 32)
        self.attention_depth = self._model_config.get('attention_depth', 3)
        self.segmentation_depth = self._model_config.get('segmentation_depth', 3)
        self.attention_hidden = self._model_config.get('attention_hidden', 32)
        self.segmentation_hidden = self._model_config.get('segmentation_hidden', 32)
        self.N = self._model_config.get('N', 3)
        self.instance_decoder_depth = self._model_config.get('instance_decoder_depth', 3)
        self.train = self.model_config.get('train', True)

        # TODO: Give option to use ResNet Blocks insteaed of Conv+BN+LeakyReLU Blocks

        # Backbone Feature Extraction Network
        self.net = UResNet(cfg, name='uresnet_lonely')
        self.num_classes = self.model_config.get('num_classes', 5)

        # Attention Proposal Branch
        self.attention_net = scn.Sequential()
        for i in range(self.attention_depth):
            module = scn.Sequential()
            module.add(
                scn.SubmanifoldConvolution(self.net._dimension,
                (self.feature_size if i == 0 else self.attention_hidden),
                self.attention_hidden, 3, self.allow_bias)).add(
                scn.BatchNormLeakyReLU(self.attention_hidden, leakiness=self.leakiness))
            self.attention_net.add(module)
        self.attention_net.add(scn.NetworkInNetwork(self.attention_hidden, 1, self.allow_bias))

        # Segmentation Branch
        self.segmentation_net = scn.Sequential()
        for i in range(self.segmentation_depth):
            module = scn.Sequential()
            module.add(
                scn.SubmanifoldConvolution(self.net._dimension,
                (self.feature_size if i == 0 else self.segmentation_hidden),
                self.segmentation_hidden, 3, self.allow_bias)).add(
                scn.BatchNormLeakyReLU(self.segmentation_hidden, leakiness=self.leakiness))
            self.segmentation_net.add(module)
        self.segmentation_net.add(scn.NetworkInNetwork(self.attention_hidden, self.num_classes, self.allow_bias))

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
                    feature_size, feature_size, 3, self.allow_bias)).add(
                scn.BatchNormLeakyReLU(feature_size, leakiness=self.leakiness))
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
                    instance_downsample[i], instance_downsample[i+1], 3, self.allow_bias)).add(
                scn.BatchNormLeakyReLU(instance_downsample[i+1], leakiness=self.leakiness))
            self.instance_dec.add(module)
        # Final Mask is Binary
        self.instance_dec.add(scn.NetworkInNetwork(instance_downsample[-1], 1, self.allow_bias))

        # OutputLayers
        self.segment_output = scn.OutputLayer(self.dimension)
        self.ppn_output = scn.OutputLayer(self.dimension)
        self.instance_output = scn.OutputLayer(self.dimension)

        # Misc
        self.mask_tensor = MaskSparseTensor(self.dimension)

    @staticmethod
    def find_query_points(coords, ppn_scores, max_points=100):
        '''
        TODO:
        Based on PPN Output, find query points to be passed to 
        AdaIN layers via local maximum finding.

        NOTE: Only used in inference. 
        '''
        return


    @staticmethod
    def find_centroids(features, labels):
        '''
        For a given image, compute the centroids mu_c for each
        cluster label in the embedding space.
        Inputs:
            features (torch.Tensor): the pixel embeddings, shape=(N, d) where
            N is the number of pixels and d is the embedding space dimension.
            labels (torch.Tensor): ground-truth group labels, shape=(N, )
        Returns:
            centroids (torch.Tensor): (n_c, d) tensor where n_c is the number of
            distinct instances. Each row is a (1,d) vector corresponding to
            the coordinates of the i-th centroid.
        '''
        clabels = labels.unique(sorted=True)
        centroids = []
        for c in clabels:
            index = (labels == c)
            mu_c = features[index].mean(0)
            centroids.append(mu_c)
        centroids = torch.stack(centroids)
        return centroids


    def find_nearest_features(self, features, coords, points):
        '''
        Given a PPN Truth point (x0, y0, z0, b0, c0), locates the
        nearest voxel in the input image. We construct a KDTree with 
        <points> and query <coords> for fast nearest-neighbor search. 

        NOTE: that PPN Truth gives a floating point coordinate, and the output
        feature tensors have integer spatial coordinates of original space.

        NOTE: This function should only be used in TRAINING AdaptIS. 

        INPUTS:
            - coords (N x 5 Tensor): coordinates (including batch and class)
            for the current event (fixed batch index).
            - points (N_p x 5 Tensor): PPN points to query nearest neighbor.
            Here, N_p is the number of PPN ground truth points. 

        RETURNS:
            - nearest_neighbor (1 x 5 Tensor): nearest neighbor of <point>
        '''
        with torch.no_grad():
            localCoords = coords[:, :3].detach().cpu().numpy()
            localPoints = points[:, :3].detach().cpu().numpy()
            tree = cKDTree(localPoints)
            dists, indices = tree.query(localCoords, k=1,
                             distance_upper_bound=self.spatial_size)
            perm = np.argsort(dists)
            _, indices = np.unique(indices[perm], return_index=True)
        return features[perm[indices]]


    def train_loop(self, features, query_points, segment_label, cluster_label):
        '''
        Training loop for AdaptIS
        '''
        coords = features.get_spatial_locations()
        batch_idx = coords[:, -1].int().unique()
        ppn_points = query_points[0]
        slabels = segment_label[0]
        clabels = cluster_label[0]
        pred_logits, pred_coords = [], []
        for i, bidx in enumerate(batch_idx):
            batch_mask = coords[:, 3] == bidx
            points_batch = ppn_points[ppn_points[:, 3] == bidx]
            coords_batch = coords[batch_mask]
            slabels_batch = slabels[batch_mask]
            clabels_batch = clabels[batch_mask]
            self.mask_tensor.mask = batch_mask
            features_batch = self.mask_tensor(features)
            #feature_batch = feature_tensor[batch_mask]
            semantic_classes = slabels_batch[:, -1].unique()
            for c in semantic_classes:
                class_mask = slabels_batch[:, -1] == c
                coords_class = coords_batch[class_mask]
                self.mask_tensor.mask = class_mask
                features_class = self.mask_tensor(features_batch)
                #features_class = feature_batch[class_mask]
                clabels_class = clabels_batch[class_mask]
                if c == 2:
                    points_class = points_batch[points_batch[:, -1] == c]
                else:
                    points_class = self.find_centroids(coords_class[:, :3], clabels_class)
                sampled_features = self.find_nearest_features(
                    features_class.features, coords_class, points_class)
                mask_class, mask_coords = [], []
                for sample in sampled_features:
                    weight = self.controller_weight(sample)
                    bias = self.controller_bias(sample)
                    self.adapt_in.set_params(weight, bias)
                    x = self.instance_net(features_class)
                    x = self.adapt_in(x)
                    x = self.instance_dec(x) # Mask features, later apply BCE.
                    mask_class.append(x.features)
                    mask_coords.append(x.get_spatial_locations())
                mask_class = torch.cat(mask_class, dim=0)
                mask_coords = torch.cat(mask_coords, dim=0)
            pred_logits.append(mask_class)
            pred_coords.append(mask_coords)
        pred_logits = torch.cat(pred_logits, dim=0)
        pred_coords = torch.cat(pred_coords, dim=0)
        perm = np.lexsort(pred_coords[:, 2], pred_coords[:, 1], 
                            pred_coords[:, 0], pred_coords[:, 3])
        pred_coords = pred_coords[perm]
        pred_logits = pred_logits[perm]
        logits = scn.InputBatch(self.dimension, self.spatial_size)
        logits.set_locations(pred_coords)
        logits.features = pred_logits
        return logits


    def test_loop(self, features, ppn_scores):
        '''
        Inference loop for AdaptIS
        '''
        coords = features.get_spatial_locations()
        batch_idx = coords[:, -1].int().unique()
        segmented = torch.zeros(features.features.shape[0])
        for i, bidx in enumerate(batch_idx):
            batch_mask = coords[:, 3] == bidx
            coords_batch = coords[batch_mask]
            ppn_batch = ppn_scores[batch_mask]
            self.mask_tensor.mask = batch_mask
            features_batch = self.mask_tensor(features)

        
    def forward(self, input, query_points=None, segment_label=None, cluster_label=None):
        '''
        INPUTS:
            - input: usual input to UResNet
            - query_points: PPN Truth (only used during training)

            During training, we sample random points at least once from each cluster.
            During inference, we sample the highest attention scoring points.
        
        TODO: Turn off attention map training. 
        '''
        TRACK_LABELS = set([0,1])
        net_output = self.net(input)
        # Get last feature layer in UResNet
        features = net_output['features_dec'][0][-1]

        # Point Proposal map and Segmentation Map is Holistic. 
        ppn_scores = self.attention_net(features)
        segmentation_scores = self.segmentation_net(features)

        # For Instance Branch, mask generation is instance-by-instance.
        if self.train:
            instance_scores = self.train_loop(features, query_points, segment_label, cluster_label)
        else:
            instance_scores = self.test_loop(features)

        # Return Logits for Cross-Entropy Loss
        res = {
            'segmentation': [segmentation_scores],
            'ppn': [ppn_scores],
            'instance_scores': [instance_scores]
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
