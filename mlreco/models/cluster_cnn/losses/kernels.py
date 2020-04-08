import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

# All kernel functions should produce values between [0,1], where
# similar features have kernel values closer to 1.


def gauss(centroid, sigma=1, eps=1e-8):
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        p = torch.exp(-dists / (2 * (sigma**2) + eps))
        return p
    return f

def rational_quadratic(centroid, alpha=1, eps=1e-8):
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        p = torch.pow(1.0 + dists / (2 * alpha), -alpha)
        return p
    return f

def cosine_similarity(centroid, eps=1e-8):
    def f(x):
        cent = centroid.expand_as(x)
        dists = F.cosine_similarity(x, cent)
        return (1.0 + dists) / 2
    return f

def multi_gauss(centroid, sigma, Lprime, eps=1e-8):
    def f(x):
        N = x.shape[0]
        L = torch.zeros(3, 3).cuda()
        tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
        L[tril_indices[0], tril_indices[1]] = 1 / (Lprime + eps)
        L += torch.diag(1 / (sigma + eps))
        precision = torch.matmul(L, L.t())
        dist = torch.matmul((x - centroid), precision)
        dist = torch.bmm(dist.view(N, 1, -1), (x-centroid).view(N, -1, 1)).squeeze()
        # print(dist)
        probs = torch.clamp(torch.exp(-dist), min=eps, max=1-eps)
        # print(probs)
        return probs
    return f
