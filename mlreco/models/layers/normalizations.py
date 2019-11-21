import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import sparseconvnet as scn

class _Normalization(nn.Module):
    '''
    Abstract Base Class for normalization layers.

    References:
        - Pytorch BatchNorm Implementation:

    '''
    def __init__(self, num_features, dimension=3, eps=1e-5,
                 momentum=0.1, affine=True, track_running_stats=True):
        super(_Normalization, self).__init__()
        self.num_features = num_features
        self.dimension = dimension
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', 
                torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)


class SparseInstanceNorm(_Normalization):
    '''
    Instance Normalization Layer for Sparse Tensors.

    References:
        - Pytorch BatchNorm Implementation:

        - Instance Norm Paper:


    INPUTS:
        - x (scn.SparseConvnetTensor)
    
    RETURNS:
        - out (scn.SparseConvnetTensor)
    '''
    def __init__(self, *args, **kwargs):
        super(SparseInstanceNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        out = scn.InputBatch(self.dimension, x.spatial_size)
        coords = x.get_spatial_locations()
        out.set_locations(coords, torch.Tensor(coords.shape))
        f = x.features
        f_norm = (f - torch.mean(f, dim=0)) / torch.sqrt(
            (torch.var(f, dim=0) + self.eps))
        if self.affine:
            f_norm = self.weight * f_norm + self.bias
        out.features = f_norm
        return out


class SparseGroupNorm(_Normalization):
    '''
    Group Normalization layer for Sparse Tensors.

    References:
        - Pytorch BatchNorm Implementation:

        - Group Norm Paper: 

    INPUTS:
        - x (scn.SparseConvnetTensor)

    RETURNS:
        - out (scn.SparseConvnetTensor)
    '''
    def __init__(self, nGroups, *args, **kwargs):
        super(SparseGroupNorm, self).__init__(*args, **kwargs)
        assert (self.num_features % nGroups == 0)
        self.nGroups = nGroups
        self.f_per_group = self.num_features // nGroups

    def forward(self, x):
        out = scn.InputBatch(self.dimension, x.spatial_size)
        coords = x.get_spatial_locations()
        out.set_locations(coords, torch.Tensor(coords.shape))
        f = x.features.view(-1, self.nGroups, self.f_per_group)
        mean = f.mean(dim=[0,2], keepdim=True)
        var = f.var(dim=[0,2], keepdim=True)
        f_norm = (f - mean) / (var + self.eps).sqrt()
        f_norm = f_norm.view(-1, self.num_features)
        if self.affine:
            f_norm = self.weight * f_norm + self.bias
        out.features = f_norm
        return out


class SparsePixelNorm(nn.Module):
    '''
    Pixel Normalization Layer for Sparse Tensors.
    PixelNorm layers were used in NVIDIA's ProGAN.

    This layer normalizes the feature vector in each
    pixel to unit length, and has no trainable weights.

    References:
        - NVIDIA ProGAN: https://arxiv.org/pdf/1710.10196.pdf
    '''
    def __init__(self, num_features, dimension=3, eps=1e-5):
        super(SparsePixelNorm, self).__init__()
        self.num_features = num_features
        self.dimension = dimension
        self.eps = eps

    def forward(self, x):
        out = scn.InputBatch(self.dimension, x.spatial_size)
        coords = x.get_spatial_locations()
        out.set_locations(coords, torch.Tensor(coords.shape))
        f = x.features
        norm = torch.sum(torch.pow(f, 2), dim=1, keepdim=True)
        f_norm = f / (norm + self.eps).sqrt()
        out.features = f_norm
        return out
