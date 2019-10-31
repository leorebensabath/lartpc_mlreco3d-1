import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from .cluster_cnn.utils import add_normalized_coordinates
from .cluster_cnn.loss import DiscriminativeLoss
from mlreco.models.uresnet import UResNet

###########################################################
#
# Define one multilayer model to incorporate all options.
#
# Embedding Transforming Convolutions are added on top of 
# backbone decoder features. 
# 
# Distance Estimation Map is added on top of final layer of
# backbone decoder concatenated with final layer of clustering. 
#
###########################################################

class ClusterCNN(nn.Module):
    pass

class ClusteringLoss(nn.Module):
    pass