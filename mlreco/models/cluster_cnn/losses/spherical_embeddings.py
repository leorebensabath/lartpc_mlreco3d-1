import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from .lovasz import mean, lovasz_hinge_flat, StableBCELoss, iou_binary
from .misc import FocalLoss, WeightedFocalLoss, SoftF1Loss, SoftDiceLoss
from .kernels import *
from collections import defaultdict
import pprint


class MaskEmbeddingLoss(nn.Module):
    '''
    Loss function for Sparse Spatial Embeddings Model, with fixed
    centroids and symmetric gaussian kernels.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(MaskEmbeddingLoss, self).__init__()
        self.loss_config = cfg[name]
        self.seediness_weight = self.loss_config.get('seediness_weight', 1.0)
        self.embedding_weight = self.loss_config.get('embedding_weight', 10.0)
        self.smoothing_weight = self.loss_config.get('smoothing_weight', 1.0)
        self.spatial_size = self.loss_config.get('spatial_size', 512)

        self.kernel = self.loss_config.get('kernel', 'cosine')
        if self.kernel == 'cosine':
            self.kernel = cosine_similarity
        elif self.kernel == 'rational':
            self.kernel = rational_quadratic
        elif self.kernel == 'gauss':
            self.kernel = gauss
        else:
            raise ValueError('Invalid kernel function name!')
        # L2 Loss for Seediness and Smoothing
        self.l1loss = torch.nn.L1Loss(reduction='mean')
        self.ally_margin = self.loss_config.get('ally_margin', 1.0)
        self.enemy_margin = self.loss_config.get('enemy_margin', 0.0)
        self.ally_weight = self.loss_config.get('ally_weight', 1.0)
        self.enemy_weight = self.loss_config.get('enemy_weight', 1.0)
        self.inter_margin = self.loss_config.get('inter_margin', 0.0)
        self.inter_weight = self.loss_config.get('inter_weight', 1.0)


    def find_cluster_means(self, features, labels):
        '''
        For a given image, compute the centroids mu_c for each
        cluster label in the embedding space.
        Inputs:
            features (torch.Tensor): the pixel embeddings, shape=(N, d) where
            N is the number of pixels and d is the embedding space dimension.
            labels (torch.Tensor): ground-truth group labels, shape=(N, )
        Returns:
            cluster_means (torch.Tensor): (n_c, d) tensor where n_c is the number of
            distinct instances. Each row is a (1,d) vector corresponding to
            the coordinates of the i-th centroid.
        '''
        clabels = labels.unique(sorted=True)
        cluster_means = []
        for c in clabels:
            index = (labels == c)
            mu_c = features[index].mean(0)
            cluster_means.append(mu_c)
        cluster_means = torch.stack(cluster_means)
        return cluster_means


    def inter_cluster_loss(self, cluster_means, margin=0.0):
        '''
        Implementation of distance loss in Discriminative Loss.
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): the magnitude of the margin delta_d in the paper.
            Think of it as the distance between each separate clusters in
            embedding space.
        Returns:
            inter_loss (float): computed cross-centroid distance loss (see paper).
            Factor of 2 is included for proper normalization.
        '''
        inter_loss = 0.0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            # Inter-cluster loss is zero if there only one instance exists for
            # a semantic label.
            return 0.0
        else:
            for i, c1 in enumerate(cluster_means):
                for j, c2 in enumerate(cluster_means):
                    if i != j:
                        kernel = self.kernel(c1)
                        dist = torch.clamp(kernel(c2) - margin, min=0)
                        inter_loss += dist
            inter_loss /= float((n_clusters - 1) * n_clusters)
            return inter_loss


    def compute_embedding_loss(self, embeddings, labels):
        '''
        Computes feature embedding loss using kernel trick clustering.
        '''
        loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        inter_loss = self.inter_cluster_loss(centroids, margin=self.inter_magin)

        for i, c in enumerate(cluster_labels):
            index = labels == c
            kernel = self.kernel(centroids[i])
            ally_dist = kernel(embeddings[index])
            ally_loss += torch.clamp(self.ally_margin - ally_dist, min=0)
            if index.all():
                continue
            enemy_dist = kernel(embeddings[~index])
            enemy_loss += torch.clamp(enemy_dist - self.enemy_margin, min=0)
        loss = self.ally_weight * ally_loss + self.enemy_weight * enemy_loss
        loss += self.inter_weight * inter_loss
        return loss


    def regularization(self, cluster_means):
        '''
        Implementation of regularization loss in Discriminative Loss
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
        Returns:
            reg_loss (float): computed regularization loss (see paper).
        '''
        reg_loss = 0.0
        n_clusters, _ = cluster_means.shape
        for i in range(n_clusters):
            reg_loss += torch.norm(cluster_means[i] + 1e-8, p=2)
        reg_loss /= float(n_clusters)
        return reg_loss


    def compute_mask_loss(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        reg_loss = self.regularization(centroids)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        acc = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1.0
            mask[~index] = 0.0
            sigma = torch.mean(margins[index], dim=0)
            kernel = gauss(centroids[i], sigma=sigma)
            p = kernel(embeddings)
            probs[index] = p[index]
            loss += lovasz_hinge_flat(p, mask)
            acc += iou_binary(p > 0.5, mask)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters
        loss += reg_loss

        return loss, smoothing_loss, probs, acc


    def mask_loss_class(self, embeddings, margins, seediness, slabels, clabels, coords):
        '''
        Wrapper function for combining different components of the loss,
        in particular when clustering must be done PER SEMANTIC CLASS.

        NOTE: When there are multiple semantic classes, we compute the DLoss
        by first masking out by each semantic segmentation (ground-truth/prediction)
        and then compute the clustering loss over each masked point cloud.

        INPUTS:
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class.
            loss[i] = computed DLoss for semantic class <i>.
            acc_segs (list): list of computed clustering accuracy for each semantic class.
        '''
        loss = defaultdict(list)
        accuracy = defaultdict(float)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            mask_loss, smoothing_loss, probs, acc = self.compute_mask_loss(
                embeddings[index], margins[index], clabels[index], coords[index])
            prob_truth = probs.detach()
            seed_loss = self.l1loss(prob_truth, seediness[index].squeeze(1))
            total_loss = self.embedding_weight * mask_loss \
                       + self.seediness_weight * seed_loss \
                       + self.smoothing_weight * smoothing_loss
            loss['loss'].append(total_loss)
            loss['mask_loss'].append(float(self.embedding_weight * mask_loss))
            loss['seed_loss'].append(float(self.seediness_weight * seed_loss))
            loss['smoothing_loss'].append(float(self.smoothing_weight * smoothing_loss))
            loss['mask_loss_{}'.format(int(sc))].append(float(mask_loss))
            loss['seed_loss_{}'.format(int(sc))].append(float(seed_loss))
            accuracy['accuracy_{}'.format(int(sc))] = acc

        return loss, accuracy


    def embedding_loss_class(self, embeddings, slabels, clabels):
        '''
        Wrapper function for summing feature embedding loss across classes.
        '''
        loss = 0
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            embedding_loss = self.compute_embedding_loss(
                embeddings[index], clabels[index])
            loss += embedding_loss
        loss /= semantic_classes.shape[0]

        return loss


    def forward(self, out, segment_labels, group_labels):

        num_gpus = len(segment_labels)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i_gpu in range(num_gpus):
            slabels = segment_labels[i_gpu]
            clabels = group_labels[i_gpu]
            print(slabels)
            print(clabels)
            assert False
            slabels = slabels.int()
            clabels = group_label[i][:, -2]
            batch_idx = segment_label[i][:, 3]
            embedding = out['embeddings'][i]
            seediness = out['seediness'][i]
            margins = out['margins'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]
                seed_batch = seediness[batch_idx == bidx]
                margins_batch = margins[batch_idx == bidx]
                coords_batch = coords[batch_idx == bidx] / self.spatial_size

                loss_class, acc_class = self.combine_multiclass(
                    embedding_batch, margins_batch,
                    seed_batch, slabels_batch, clabels_batch, coords_batch)
                for key, val in loss_class.items():
                    loss[key].append(sum(val) / len(val))
                for s, acc in acc_class.items():
                    accuracy[s].append(acc)
                acc = sum(acc_class.values()) / len(acc_class.values())
                accuracy['accuracy'].append(acc)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)
        # pprint.pprint(res)

        return res
