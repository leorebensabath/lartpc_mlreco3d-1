import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn


def gauss(centroid, sigma, eps=1e-8):
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        p = torch.clamp(torch.exp(-dists / (2 * torch.pow(sigma, 2))), min=0, max=1)
        return probs
    return f

def rational_quadratic(centroid, sigma=1, eps=1e-8):
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        p = torch.clamp(torch.pow(1.0 + dists / (2 * sigma), -sigma), min=0, max=1)
        return probs
    return f

def cosine_similarity(centroid, eps=1e-8):
    def f(x):
        dists = F.cosine_similarity(x, centroid, dim=1)
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
