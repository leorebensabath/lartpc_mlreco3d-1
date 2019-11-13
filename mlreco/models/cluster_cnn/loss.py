import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sparseconvnet as scn

from collections import defaultdict
from mlreco.utils.utils import ForwardData
from .utils import distance_matrix, pairwise_distances


class DiscriminativeLoss(torch.nn.Module):
    '''
    Implementation of the Discriminative Loss Function in Pytorch.
    https://arxiv.org/pdf/1708.02551.pdf
    Note that there are many other implementations in Github, yet here
    we tailor it for use in conjuction with Sparse UResNet.
    '''

    def __init__(self, cfg, reduction='sum'):
        super(DiscriminativeLoss, self).__init__()
        self.loss_config = cfg['modules']['clustering_loss']
        self.num_classes = self.loss_config.get('num_classes', 5)
        self.depth = self.loss_config.get('stride', 5)

        # Clustering Loss Parameters
        self.loss_hyperparams = {}
        self.loss_hyperparams['intra_weight'] = self.loss_config.get('intra_weight', 1.0)
        self.loss_hyperparams['inter_weight'] = self.loss_config.get('inter_weight', 1.0)
        self.loss_hyperparams['reg_weight'] = self.loss_config.get('reg_weight', 0.001)
        self.loss_hyperparams['intra_margin'] = self.loss_config.get('intracluster_margin', 0.5)
        self.loss_hyperparams['inter_margin'] = self.loss_config.get('intercluster_margin', 1.5)

        self.dimension = self.loss_config.get('data_dim', 3)
        self.norm = self.loss_config.get('norm', 2)
        self.use_segmentation = self.loss_config.get('use_segmentation', True)

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

    def intra_cluster_loss(self, features, labels, cluster_means, margin=0.5):
        '''
        Implementation of variance loss in Discriminative Loss.
        Inputs:
            features (torch.Tensor): pixel embedding, same as in find_cluster_means.
            labels (torch.Tensor): ground truth instance labels
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): constant used to specify delta_v in paper. Think of it
            as the size of each clusters in embedding space. 
        Returns:
            intra_loss: (float) variance loss (see paper).
        '''
        intra_loss = 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            dists = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm,
                               dim=1)
            hinge = torch.clamp(dists - margin, min=0)
            l = torch.mean(torch.pow(hinge, 2))
            intra_loss += l
        intra_loss /= n_clusters
        return intra_loss

    def inter_cluster_loss(self, cluster_means, margin=1.5):
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
                        dist = torch.norm(c1 - c2 + 1e-8, p=self.norm)
                        hinge = torch.clamp(2.0 * margin - dist, min=0)
                        inter_loss += torch.pow(hinge, 2)
            inter_loss /= float((n_clusters - 1) * n_clusters)
            return inter_loss

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
            reg_loss += torch.norm(cluster_means[i, :] + 1e-8, p=self.norm)
        reg_loss /= float(n_clusters)
        return reg_loss

    def compute_heuristic_accuracy(self, embedding, truth):
        '''
        Compute Adjusted Rand Index Score for given embedding coordinates,
        where predicted cluster labels are obtained from distance to closest
        centroid (computes heuristic accuracy). 

        Inputs:
            embedding (torch.Tensor): (N, d) Tensor where 'd' is the embedding dimension.
            truth (torch.Tensor): (N, ) Tensor for the ground truth clustering labels.
        Returns:
            score (float): Computed ARI Score
            clustering (array): the predicted cluster labels.
        '''
        from sklearn.metrics import adjusted_rand_score
        nearest = []
        with torch.no_grad():
            cmeans = self.find_cluster_means(embedding, truth)
            for centroid in cmeans:
                dists = torch.sum((embedding - centroid)**2, dim=1)
                dists = dists.view(-1, 1)
                nearest.append(dists)
            nearest = torch.cat(nearest, dim=1)
            nearest = torch.argmin(nearest, dim=1)
            pred = nearest.cpu().numpy()
            grd = truth.cpu().numpy()
            score = adjusted_rand_score(pred, grd)
        return score

    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses. 
        intra_margin = kwargs.get('intra_margin', 0.5)
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           margin=intra_margin)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        return {
            'loss': loss, 
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss)
        }


    def combine_multiclass(self, features, slabels, clabels, **kwargs):
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
        loss, acc_segs = defaultdict(list), defaultdict(float)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            num_clusters = len(clabels[index].unique())
            loss_blob = self.combine(features[index], clabels[index], **kwargs)
            for key, val in loss_blob.items():
                loss[key].append(val)
            # loss['loss'].append(loss_blob['loss'])
            # loss['intra_loss'].append(loss_blob['intra_loss'])
            # loss['inter_loss'].append(loss_blob['inter_loss'])
            # loss['reg_loss'].append(loss_blob['reg_loss'])
            acc = self.compute_heuristic_accuracy(features[index], clabels[index])
            acc_segs['accuracy_{}'.format(sc.item())] = acc
        return loss, acc_segs

    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc. 
        '''
        num_gpus = len(semantic_labels)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i in range(num_gpus):
            slabels = semantic_labels[i][:, -1]
            slabels = slabels.int()
            clabels = group_labels[i][:, -1]
            batch_idx = semantic_labels[i][:, 3]
            embedding = out['cluster_feature'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]

                if self.use_segmentation:
                    loss_dict, acc_segs = self.combine_multiclass(
                        embedding_batch, slabels_batch, clabels_batch, **self.loss_hyperparams)
                    for key, val in loss_dict.items():
                        loss[key].append(sum(val) / len(val))
                    for s, acc in acc_segs.items():
                        accuracy[s].append(acc)
                    acc = sum(acc_segs.values()) / len(acc_segs.values())
                    accuracy['accuracy'].append(acc)
                else:
                    loss["loss"].append(self.combine(embedding_batch, clabels_batch, **self.loss_hyperparams))
                    acc, _ = self.compute_heuristic_accuracy(embedding_batch, clabels_batch)
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

        return res


class MultiScaleLoss(DiscriminativeLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(MultiScaleLoss, self).__init__(cfg)
        self.loss_config = cfg['modules']['clustering_loss']
        self.num_strides = self.loss_config.get('num_strides', 5)

        self.intra_margins = self.loss_config.get('intra_margins', 
            [self.loss_hyperparams['intra_margin'] / 2**i for i in range(self.num_strides)])
        self.inter_margins = self.loss_config.get('inter_margins',
            [self.loss_hyperparams['inter_margin'] / 2**i for i in range(self.num_strides)])
        

    def compute_loss_layer(self, embedding_scn, slabels, clabels, batch_idx, **kwargs):
        '''
        Compute the multi-class loss for a feature map on a given layer.
        We group the loss computation to a function in order to compute the
        clustering loss over the decoding feature maps.

        INPUTS:
            - embedding (torch.Tensor): (N, d) Tensor with embedding space
                coordinates.
            - slabels (torch.Tensor): (N, 5) Tensor with segmentation labels
            - clabels (torch.Tensor): (N, 5) Tensor with cluster labels
            - batch_idx (list): list of batch indices, ex. [0, 1, ..., 4]

        OUTPUT:
            - loss (torch.Tensor): scalar number (1x1 Tensor) corresponding
                to calculated loss over a given layer.
        '''
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        coords = embedding_scn.get_spatial_locations().numpy()
        perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
        embedding = embedding_scn.features[perm]
        coords = coords[perm]

        for bidx in batch_idx:
            index = slabels[:, 3].int() == bidx
            embedding_batch = embedding[index]
            slabels_batch = slabels[index][:, -1]
            clabels_batch = clabels[index][:, -1]
            # Compute discriminative loss for current event in batch
            if self.use_segmentation:
                loss_dict, acc_segs = self.combine_multiclass(
                    embedding_batch, slabels_batch, clabels_batch, **kwargs)
                for key, val in loss_dict.items():
                    loss[key].append( sum(val) / float(len(val)) )
                for s, acc in acc_segs.items():
                    accuracy[s].append(acc)
                acc = sum(acc_segs.values()) / len(acc_segs.values())
                accuracy['accuracy'].append(acc)
            else:
                loss["loss"].append(self.combine(embedding_batch, clabels_batch, **kwargs))
                acc = self.compute_heuristic_accuracy(embedding_batch, clabels_batch)
                accuracy['accuracy'].append(acc)

        # Averaged over batch at each layer
        loss = { key : sum(l) / float(len(l)) for key, l in loss.items() }
        accuracy = { key : sum(l) / float(len(l)) for key, l in accuracy.items() }
        return loss, accuracy


    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc.
        '''

        loss = defaultdict(list)
        accuracy = defaultdict(list)
        num_gpus = len(semantic_labels)
        num_layers = len(out['cluster_feature'][0])

        for i_gpu in range(num_gpus):
            batch_idx = semantic_labels[i_gpu][0][:, 3].detach().cpu().int().numpy()
            batch_idx = np.unique(batch_idx)
            batch_size = len(batch_idx)
            # Summing clustering loss over layers.
            for i, em in enumerate(out['cluster_feature'][i_gpu]):
                delta_var, delta_dist = self.intra_margins[i], self.inter_margins[i]
                loss_i, acc_i = self.compute_loss_layer(
                    em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                    delta_var=delta_var, delta_dist=delta_dist)
                for key, val in loss_i.items():
                    loss[key].append(val)
                # Compute accuracy only at last layer.
                if i == 0:
                    acc_clustering = acc_i
            for key, acc in acc_clustering.items():
                # Batch Averaged Accuracy
                accuracy[key].append(acc)

        # Average over layers and num_gpus
        loss_avg = {}
        acc_avg = {}
        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


class MultiScaleLoss2(MultiScaleLoss):
    '''
    Same as multi scale loss, but we include enemy loss in intra loss.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(MultiScaleLoss2, self).__init__(cfg, name=name)
        self.ally_weight = self.loss_config.get('ally_weight', 1.0)
        self.ally_margins = self.intra_margins
        self.enemy_weight = self.loss_config.get('enemy_weight', 10.0)
        self.loss_hyperparams['enemy_margin'] = self.loss_config.get('enemy_margin', 1.0)
        self.enemy_margins = self.loss_config.get('enemy_margins',
            [self.loss_hyperparams['enemy_margin'] / 2**i for i in range(self.num_strides)])


    def intra_cluster_loss(self, features, labels, cluster_means,
                           ally_margin=0.5, enemy_margin=1.0):
        '''
        Intra-cluster loss, with per-voxel weighting and enemy loss.
        This variant of intra-cluster loss penalizes the distance 
        from the centroid to its enemies in addition to pulling 
        ally points towards the center. 

        INPUTS:
            - ally_margin (float): centroid pulls all allied points
            inside this margin.
            - enemy_margin (float): centroid pushs all enemy points
            inside this margin.
            - weight: 
        '''
        intra_loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm, dim=1)
            allies = torch.clamp(allies - ally_margin, min=0)
            x = self.ally_weight * torch.mean(torch.pow(allies, 2))
            intra_loss += x
            ally_loss += float(x)
            if index.all():
                continue
            enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
                    p=self.norm, dim=1)
            enemies = torch.clamp(enemy_margin - enemies, min=0)
            x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
            intra_loss += x
            enemy_loss += float(x)

        intra_loss /= n_clusters
        ally_loss /= n_clusters
        enemy_loss /= n_clusters
        return intra_loss, ally_loss, enemy_loss


    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses. 
        ally_margin = kwargs.get('ally_margin', 0.5)
        enemy_margin = kwargs.get('enemy_margin', 1.0)
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss, ally_loss, enemy_loss = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           ally_margin=ally_margin,
                                           enemy_margin=enemy_margin)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        return {
            'loss': loss, 
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss),
            'ally_loss': intra_weight * float(ally_loss),
            'enemy_loss': intra_weight * float(enemy_loss)
        }


    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc.
        '''

        loss = defaultdict(list)
        accuracy = defaultdict(list)
        num_gpus = len(semantic_labels)
        num_layers = len(out['cluster_feature'][0])

        for i_gpu in range(num_gpus):
            batch_idx = semantic_labels[i_gpu][0][:, 3].detach().cpu().int().numpy()
            batch_idx = np.unique(batch_idx)
            batch_size = len(batch_idx)
            # Summing clustering loss over layers.
            for i, em in enumerate(out['cluster_feature'][i_gpu]):
                delta_var, delta_dist = self.ally_margins[i], self.inter_margins[i]
                delta_enemy = self.enemy_margins[i]
                loss_i, acc_i = self.compute_loss_layer(
                    em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                    ally_margin=delta_var, inter_margin=delta_dist, enemy_margin=delta_enemy)
                for key, val in loss_i.items():
                    loss[key].append(val)
                # Compute accuracy only at last layer.
                if i == 0:
                    acc_clustering = acc_i
            for key, acc in acc_clustering.items():
                # Batch Averaged Accuracy
                accuracy[key].append(acc)

        # Average over layers and num_gpus
        loss_avg = {}
        acc_avg = {}
        for key, val in loss.items():
            loss_avg[key] = sum(val) / len(val)
        for key, val in accuracy.items():
            acc_avg[key] = sum(val) / len(val)

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


class WeightedMultiLoss(MultiScaleLoss):
    '''
    Same as MultiScaleLoss, but with attention weighting.
    '''
    def __init__(self, cfg, name='clustering_loss'):
        super(WeightedMultiLoss, self).__init__(cfg, name=name)
        self.attention_kernel = self.loss_config.get('attention_kernel', 1)
        if self.attention_kernel == 0:
            self.kernel_func = lambda x: 1.0 + torch.exp(-x)
        elif self.attention_kernel == 1:
            self.kernel_func = lambda x: 2.0 / (1 + torch.exp(-x))
        else:
            raise ValueError('Invalid weighting kernel function mode.')


    def intra_cluster_loss(self, features, labels, cluster_means, coords, margin=0.5):
        '''
        Intra-cluster loss, with per-voxel weighting and enemy loss.
        This variant of intra-cluster loss penalizes the distance 
        from the centroid to its enemies in addition to pulling 
        ally points towards the center. 

        INPUTS:
            - ally_margin (float): centroid pulls all allied points
            inside this margin.
            - enemy_margin (float): centroid pushs all enemy points
            inside this margin.
            - weight: Tensor with features.shape[0] entries corresponding to
            attention weights. 
        '''
        intra_loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        with torch.no_grad():
            coords_mean = self.find_cluster_means(coords, labels)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm, dim=1)
            allies = torch.clamp(allies - margin, min=0)
            with torch.no_grad():
                dists = torch.norm(coords[index] - coords_mean[i] + 1e-8,
                                    p=self.norm, dim=1)
                dists = dists / (dists.std(unbiased=False) + 1e-4)
                weight = self.kernel_func(dists)
            intra_loss += torch.mean(weight * torch.pow(allies, 2))

        intra_loss /= n_clusters
        return intra_loss


    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses. 
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_margin = kwargs.get('intra_margin', 0.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)
        coords = kwargs['coords']

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           coords,
                                           margin=intra_margin)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        return {
            'loss': loss, 
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss)
        }


    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        num_gpus = len(segment_label)
        loss, accuracy = defaultdict(list), defaultdict(list)

        # Loop first over scaled feature maps
        for i_gpu in range(num_gpus):
            for depth in range(self.depth):

                batch_ids = segment_label[i_gpu][depth][:, 3].detach().cpu().int().numpy()
                batch_ids = np.unique(batch_ids)
                batch_size = len(batch_ids)

                embedding = result['cluster_feature'][i_gpu][depth]
                clabels_depth = cluster_label[i_gpu][depth]
                slabels_depth = segment_label[i_gpu][depth]

                coords = embedding.get_spatial_locations()[:, :4]
                perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
                coords = coords[perm].float().cuda()
                feature_map = embedding.features[perm]

                loss_event = defaultdict(list)
                acc_event = defaultdict(list)
                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    loss_class = defaultdict(list)
                    acc_class = defaultdict(list)
                    acc_avg = 0.0
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -1]
                        coords_class = coords_event[class_mask][:, :3]
                        # Clustering Loss
                        if depth == 0:
                            acc = self.compute_heuristic_accuracy(embedding_class,
                                                                cluster_class)
                            accuracy['accuracy_{}'.format(k)].append(acc)
                            acc_avg += acc / n_classes
                        closs = self.combine(embedding_class,
                                             cluster_class,
                            intra_margin=self.intra_margins[depth],
                            inter_margin=self.inter_margins[depth],
                            coords=coords_class)
                        for key, val in closs.items():
                            loss_class[key].append(val)
                    if depth == 0:
                        accuracy['accuracy'].append(acc_avg)
                    for key, val in loss_class.items():
                        loss_event[key].append(sum(val) / len(val))
                for key, val in loss_event.items():
                    loss[key].append(sum(val) / len(val))
                    
        res = {}

        for key, val in loss.items():
            res[key] = sum(val) / len(val)

        for key, val in accuracy.items():
            res[key] = sum(val) / len(val)

        return res



class NeighborLoss(MultiScaleLoss):
    '''
    Distance to Neighboring Ally and Enemy Loss

    NOTE: This function has HUGE memory footprint and training
    will crash under current implementation.
    '''
    def __init__(self, cfg):
        super(NeighborLoss, self).__init__(cfg)
        self.loss_config = cfg['modules']['clustering_loss']

        # Huber Loss for Team Loss
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='mean')

        # Density Loss Parameters
        self.estimate_teams = self.loss_config.get('estimate_teams', False)
        self.ally_est_weight = self.loss_config.get('ally_est_weight', 1.0)
        self.enemy_est_weight = self.loss_config.get('enemy_est_weight', 1.0)

        # Minimum Required Distance^2 to Closest Ally
        self.targetAlly = self.loss_config.get('target_friends', 1.0)
        # Maximum Required Distance^2 to Closest Enemy
        self.targetEnemy = self.loss_config.get('target_enemies', 5.0)

        self.ally_margins = self.loss_config.get('ally_margins', 
            [self.targetAlly / 2**i for i in range(self.num_strides)])
        self.enemy_margins = self.loss_config.get('enemy_margins',
            [self.targetEnemy / 2**i for i in range(self.num_strides)])

        self.ally_weight = self.loss_config.get('ally_weight', 1.0)
        self.enemy_weight = self.loss_config.get('enemy_weight', 1.0)
        self.affinity_weight = self.loss_config.get('affinity_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)


    def compute_neighbor_loss(self, embedding_class, cluster_class,
            ally_margin=0.25, enemy_margin=10.0):
        """
        Computes voxel team loss.

        INPUTS:
            (torch.Tensor)
            - embedding_class: class-masked hyperspace embedding
            - cluster_class: class-masked cluster labels

        RETURNS:
            - loss (torch.Tensor): scalar tensor representing aggregated loss.
            - dlossF (dict of floats): dictionary of ally loss.
            - dlossE (dict of floats): dictionary of enemy loss.
            - dloss_map (torch.Tensor): computed ally/enemy affinity for each voxel. 
        """
        loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        dist = distance_matrix(embedding_class)
        cluster_ids = cluster_class.unique().int()
        num_clusters = float(cluster_ids.shape[0])
        for c in cluster_ids:
            index = cluster_class.int() == c
            allies = dist[index, :][:, index]
            num_allies = allies.shape[0]
            if num_allies <= 1:
                # Skip if only one point
                continue
            ind = np.diag_indices(num_allies)
            allies[ind[0], ind[1]] = float('inf')
            allies, _ = torch.min(allies, dim=1)
            lossA = self.ally_weight *  torch.mean(
                torch.clamp(allies - ally_margin, min=0))
            loss += lossA
            ally_loss += float(lossA)
            del lossA
            if index.all(): 
                # Skip if there are no enemies
                continue
            enemies, _ = torch.min(dist[index, :][:, ~index], dim=1)
            lossE = self.enemy_weight * torch.mean(
                torch.clamp(enemy_margin - enemies, min=0))
            loss += lossE
            enemy_loss += float(lossE)
            del lossE
        
        loss /= num_clusters
        ally_loss /= num_clusters
        enemy_loss /= num_clusters
        return loss, ally_loss, enemy_loss


    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        data = ForwardData()
        num_gpus = len(segment_label)
        loss = 0.0
        clustering_loss, affinity_loss = 0.0, 0.0
        count = 0

        # Loop first over scaled feature maps
        for i_gpu in range(num_gpus):
            for depth in range(self.depth):

                batch_ids = segment_label[i_gpu][depth][:, 3].detach().cpu().int().numpy()
                batch_ids = np.unique(batch_ids)
                batch_size = len(batch_ids)

                embedding = result['cluster_feature'][i_gpu][depth]
                clabels_depth = cluster_label[i_gpu][depth]
                slabels_depth = segment_label[i_gpu][depth]

                coords = embedding.get_spatial_locations()[:, :4]
                perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
                coords = coords[perm]
                feature_map = embedding.features[perm]

                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -1]
                        # Clustering Loss
                        acc = self.compute_heuristic_accuracy(embedding_class,
                                                              cluster_class)
                        closs = self.combine(embedding_class,
                                             cluster_class,
                            intra_margin=self.intra_margins[depth],
                            inter_margin=self.inter_margins[depth])
                        dloss, dlossF, dlossE = self.compute_neighbor_loss(
                            embedding_class, cluster_class,
                            ally_margin=self.ally_margins[depth],
                            enemy_margin=self.enemy_margins[depth])
                        # Informations to be saved in log file (loss/accuracy). 
                        data.update_mean('accuracy', acc)
                        data.update_mean('intra_loss', closs['intra_loss'])
                        data.update_mean('inter_loss', closs['inter_loss'])
                        data.update_mean('reg_loss', closs['reg_loss'])
                        data.update_mean('ally_loss', dlossF)
                        data.update_mean('enemy_loss', dlossE)
                        data.update_mean('accuracy_{}'.format(class_), acc)
                        data.update_mean('intra_loss_{}'.format(class_), closs['intra_loss'])
                        data.update_mean('inter_loss_{}'.format(class_), closs['inter_loss'])
                        data.update_mean('ally_loss_{}'.format(class_), dlossF)
                        data.update_mean('enemy_loss_{}'.format(class_), dlossE)
                        clustering_loss += self.clustering_weight * closs['loss']
                        affinity_loss += self.affinity_weight * dloss
                        count += 1

        res = data.as_dict()
        res['loss'] = (clustering_loss + affinity_loss) / count
        return res


class EnhancedEmbeddingLoss(MultiScaleLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(EnhancedEmbeddingLoss, self).__init__(cfg)
        self.spatial_size = self.loss_config.get('spatial_size', 512)
        self.ally_weight = self.loss_config.get('ally_weight', 1.0)
        self.enemy_weight = self.loss_config.get('enemy_weight', 0.0)
        self.attention_kernel = self.loss_config.get('attention_kernel', 1)
        self.compute_enemy_loss = self.loss_config.get('compute_enemy_loss', True)
        self.compute_attention_weights = self.loss_config.get('compute_attention_weights', True)
        if self.attention_kernel == 0:
            self.kernel_func = lambda x: 1.0 + torch.exp(-x)
        elif self.attention_kernel == 1:
            self.kernel_func = lambda x: 2.0 / (1 + torch.exp(-x))
        else:
            raise ValueError('Invalid weighting kernel function mode.')

    def compute_attention_weight(self, coords, labels):
        '''
        Computes the per-voxel intra-cluster loss weights from
        distances to cluster centroids in coordinate space.

        INPUTS:
            - coords (N x 2,3): spatial coordinates of N voxels
            in image space.
            - labels (N x 1): cluster labels for N voxels.

        RETURNS:
            - weights (N x 1): computed attention weights for 
            N voxels.
        '''
        with torch.no_grad():
            weights = torch.zeros(labels.shape)
            if torch.cuda.is_available():
                weights = weights.cuda()
            centroids = self.find_cluster_means(coords, labels)
            cluster_labels = labels.unique(sorted=True)
            for i, c in enumerate(cluster_labels):
                index = labels == c
                dists = torch.norm(coords[index] - centroids[i] + 1e-8,
                                    p=self.norm, dim=1) / self.spatial_size
                weights[index] = self.kernel_func(dists)
        return weights


    def intra_cluster_loss(self, features, labels, cluster_means,
                           ally_margin=0.5, enemy_margin=1.0, weight=1.0):
        '''
        Intra-cluster loss, with per-voxel weighting and enemy loss.
        This variant of intra-cluster loss penalizes the distance 
        from the centroid to its enemies in addition to pulling 
        ally points towards the center. 

        INPUTS:
            - ally_margin (float): centroid pulls all allied points
            inside this margin.
            - enemy_margin (float): centroid pushs all enemy points
            inside this margin.
            - weight: 
        '''
        intra_loss = 0.0
        ally_loss, enemy_loss = 0.0, 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            allies = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm, dim=1)
            allies = torch.clamp(allies - ally_margin, min=0)
            x = self.ally_weight * torch.mean(weight[index] * torch.pow(allies, 2))
            intra_loss += x
            ally_loss += float(x)
            if index.all() or self.compute_enemy_loss:
                continue
            enemies = torch.norm(features[~index] - cluster_means[i] + 1e-8,
                    p=self.norm, dim=1)
            enemies = torch.clamp(enemy_margin - enemies, min=0)
            x = self.enemy_weight * torch.sum(torch.pow(enemies, 2))
            intra_loss += x
            enemy_loss += x

        intra_loss /= n_clusters
        ally_loss /= n_clusters
        enemy_loss / n_clusters
        return intra_loss, ally_loss, enemy_loss


    # def inter_cluster_loss(self, cluster_means, margin=1.5):
    #     '''
    #     Inter-cluster loss, vectorized with BLAS/LAPACK distance
    #     matrix computation.
    #
    #     NOTE: This function causes NaNs during backward. 
    #     '''
    #     inter_loss = 0.0
    #     n_clusters = len(cluster_means)
    #     if n_clusters < 2:
    #         # Inter-cluster loss is zero if there only one instance exists for
    #         # a semantic label.
    #         return 0.0
    #     else:
    #         inter_loss = torch.pow(torch.clamp(2.0 * margin - \
    #             torch.sqrt(distance_matrix(cluster_means) + 1e-8), min=0), 2)
    #         inter_loss = torch.triu(inter_loss, diagonal=1)
    #         inter_loss = 2 * torch.sum(inter_loss) / float((n_clusters - 1) * n_clusters)
    #         return inter_loss


    def combine(self, features, labels, **kwargs):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        # Clustering Loss Hyperparameters
        # We allow changing the parameters at each computation in order
        # to alter the margins at each spatial resolution in multi-scale losses. 
        ally_margin = kwargs.get('ally_margin', 0.5)
        enemy_margin = kwargs.get('enemy_margin', 1.0)
        inter_margin = kwargs.get('inter_margin', 1.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)
        attention_weights = kwargs.get('attention_weights', 1.0)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=inter_margin)
        intra_loss, ally_loss, enemy_loss = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           ally_margin=ally_margin,
                                           enemy_margin=enemy_margin,
                                           weight=attention_weights)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        return {
            'loss': loss, 
            'intra_loss': intra_weight * float(intra_loss),
            'inter_loss': inter_weight * float(inter_loss),
            'reg_loss': reg_weight * float(reg_loss),
            'ally_loss': intra_weight * float(ally_loss),
            'enemy_loss': intra_weight * float(enemy_loss)
        }


    def combine_multiclass(self, features, slabels, clabels,
            attention_weight=1.0, **kwargs):
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
        loss, accuracy = defaultdict(float), defaultdict(float)
        semantic_classes = slabels.unique()
        nClasses = len(semantic_classes)
        avg_acc = 0.0
        compute_accuracy = kwargs.get('compute_accuracy', False)
        for sc in semantic_classes:
            index = (slabels == sc)
            num_clusters = len(clabels[index].unique())
            loss_blob = self.combine(features[index], clabels[index],
                attention_weight=attention_weight[index], **kwargs)
            loss['loss'] += loss_blob['loss'] / nClasses
            loss['intra_loss'] += loss_blob['intra_loss'] / nClasses
            loss['inter_loss'] += loss_blob['inter_loss'] / nClasses
            loss['reg_loss'] += loss_blob['reg_loss'] / nClasses
            loss['ally_loss'] += loss_blob['ally_loss'] / nClasses
            loss['enemy_loss'] += loss_blob['enemy_loss'] / nClasses
            if compute_accuracy:
                acc = self.compute_heuristic_accuracy(features[index], clabels[index])
                accuracy['accuracy_{}'.format(sc.item())] = acc
                avg_acc += acc / nClasses
        accuracy['accuracy'] = avg_acc
        return loss, accuracy


    def compute_loss_layer(self, embedding_scn, slabels, clabels, batch_idx, **kwargs):
        '''
        Compute the multi-class loss for a feature map on a given layer.
        We group the loss computation to a function in order to compute the
        clustering loss over the decoding feature maps.

        INPUTS:
            - embedding (torch.Tensor): (N, d) Tensor with embedding space
                coordinates.
            - slabels (torch.Tensor): (N, 5) Tensor with segmentation labels
            - clabels (torch.Tensor): (N, 5) Tensor with cluster labels
            - batch_idx (list): list of batch indices, ex. [0, 1, ..., 4]

        OUTPUT:
            - loss (torch.Tensor): scalar number (1x1 Tensor) corresponding
                to calculated loss over a given layer.
        '''
        loss = ForwardData()
        accuracy = ForwardData()

        coords = embedding_scn.get_spatial_locations()
        coords_np = coords.numpy()
        perm = np.lexsort((coords_np[:, 2], coords_np[:, 1],
                           coords_np[:, 0], coords_np[:, 3]))
        embedding = embedding_scn.features[perm]
        coords = coords[perm].float()
        if torch.cuda.is_available():
            coords = coords.cuda()

        for bidx in batch_idx:
            index = slabels[:, 3].int() == bidx
            embedding_batch = embedding[index]
            slabels_batch = slabels[index][:, -1]
            clabels_batch = clabels[index][:, -1]
            coords_batch = coords[index][:, :3]
            attention_weights = self.compute_attention_weight(coords_batch, clabels_batch)
            loss_dict, acc_dict = self.combine_multiclass(
                embedding_batch, slabels_batch, clabels_batch,
                attention_weights=attention_weights, **kwargs)
            loss.update_dict(loss_dict)
            accuracy.update_dict(acc_dict)

        return loss.as_dict(), accuracy.as_dict()


    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc.
        '''

        data = ForwardData()
        for i_gpu in range(len(semantic_labels)):
            batch_idx = semantic_labels[i_gpu][0][:, 3].detach().cpu().int().numpy()
            batch_idx = np.unique(batch_idx)
            batch_size = len(batch_idx)
            # Summing clustering loss over layers.
            for i, em in enumerate(out['cluster_feature'][i_gpu]):
                # Get scaled margins for each layer.
                delta_var, delta_dist = self.intra_margins[i], self.inter_margins[i]
                # Compute accuracy at last layer.
                if i == 0:
                    em = out['final_embedding'][i_gpu]
                    loss_i, acc_i = self.compute_loss_layer(
                        em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                        delta_var=delta_var, delta_dist=delta_dist,
                        compute_accuracy=True)
                    data.update_dict(loss_i)
                    data.update_dict(acc_i)
                else:
                    loss_i, acc_i = self.compute_loss_layer(
                        em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                        delta_var=delta_var, delta_dist=delta_dist,
                        compute_accuracy=False)
                    data.update_dict(loss_i)

        res = data.as_dict()
        return res


class DistanceEstimationLoss(MultiScaleLoss):


    def __init__(self, cfg, name='clustering_loss'):
        super(DistanceEstimationLoss, self).__init__(cfg, name='uresnet')
        self.loss_config = cfg['modules'][name]
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='mean')
        self.distance_estimate_weight = self.loss_config.get('distance_estimate_weight', 1.0)
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)

    def get_nn_map(self, embedding_class, cluster_class):
        """
        Computes voxel team loss.

        INPUTS:
            (torch.Tensor)
            - embedding_class: class-masked hyperspace embedding
            - cluster_class: class-masked cluster labels

        RETURNS:
            - loss (torch.Tensor): scalar tensor representing aggregated loss.
            - dlossF (dict of floats): dictionary of ally loss.
            - dlossE (dict of floats): dictionary of enemy loss.
            - dloss_map (torch.Tensor): computed ally/enemy affinity for each voxel. 
        """
        with torch.no_grad():
            allyMap = torch.zeros(embedding_class.shape[0])
            enemyMap = torch.zeros(embedding_class.shape[0])
            if torch.cuda.is_available():
                allyMap = allyMap.cuda()
                enemyMap = enemyMap.cuda() 
            dist = distance_matrix(embedding_class)
            cluster_ids = cluster_class.unique().int()
            num_clusters = float(cluster_ids.shape[0])
            for c in cluster_ids:
                index = cluster_class.int() == c
                allies = dist[index, :][:, index]
                num_allies = allies.shape[0]
                if num_allies <= 1:
                    # Skip if only one point
                    continue
                ind = np.diag_indices(num_allies)
                allies[ind[0], ind[1]] = float('inf')
                allies, _ = torch.min(allies, dim=1)
                allyMap[index] = allies
                if index.all(): 
                    # Skip if there are no enemies
                    continue
                enemies, _ = torch.min(dist[index, :][:, ~index], dim=1)
                enemyMap[index] = enemies

            nnMap = torch.cat([allyMap.view(-1, 1), enemyMap.view(-1, 1)], dim=1)         
            return nnMap


    def forward(self, result, segment_label, cluster_label):
        '''
        Mostly borrowed from uresnet_clustering.py
        '''
        num_gpus = len(segment_label)
        loss, accuracy = defaultdict(list), defaultdict(list)

        # Loop first over scaled feature maps
        for i_gpu in range(num_gpus):
            distance_estimate = result['distance_estimation'][i_gpu]
            dcoords = distance_estimate.get_spatial_locations()[:, :4]
            perm = np.lexsort((dcoords[:, 2], dcoords[:, 1], dcoords[:, 0], dcoords[:, 3]))
            distance_estimate = distance_estimate.features[perm]
            for depth in range(self.depth):

                batch_ids = segment_label[i_gpu][depth][:, 3].detach().cpu().int().numpy()
                batch_ids = np.unique(batch_ids)
                batch_size = len(batch_ids)

                embedding = result['cluster_feature'][i_gpu][depth]
                clabels_depth = cluster_label[i_gpu][depth]
                slabels_depth = segment_label[i_gpu][depth]

                coords = embedding.get_spatial_locations()[:, :4]
                perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
                coords = coords[perm].float().cuda()
                feature_map = embedding.features[perm]

                loss_event = defaultdict(list)
                acc_event = defaultdict(list)
                for bidx in batch_ids:
                    batch_mask = coords[:, 3] == bidx
                    hypercoordinates = feature_map[batch_mask]
                    slabels_event = slabels_depth[batch_mask]
                    clabels_event = clabels_depth[batch_mask]
                    coords_event = coords[batch_mask]
                    if depth == 0:
                        distances_event = distance_estimate[batch_mask]

                    # Loop over semantic labels:
                    semantic_classes = slabels_event[:, -1].unique()
                    n_classes = len(semantic_classes)
                    loss_class = defaultdict(list)
                    acc_class = defaultdict(list)
                    acc_avg = 0.0
                    for class_ in semantic_classes:
                        k = int(class_)
                        class_mask = slabels_event[:, -1] == class_
                        embedding_class = hypercoordinates[class_mask]
                        cluster_class = clabels_event[class_mask][:, -1]
                        coords_class = coords_event[class_mask][:, :3]
                        if depth == 0:
                            distances_class = distances_event[class_mask]
                            acc = self.compute_heuristic_accuracy(embedding_class,
                                                                cluster_class)
                            accuracy['accuracy_{}'.format(k)].append(acc)
                            acc_avg += acc / n_classes
                            dMap = self.get_nn_map(embedding_class, cluster_class)
                            dloss = self.huber_loss(dMap, distances_class) * self.distance_estimate_weight
                            loss_class['distance_estimation'].append(dloss)
                        # Clustering Loss
                        closs = self.combine(embedding_class,
                                             cluster_class,
                            intra_margin=self.intra_margins[depth],
                            inter_margin=self.inter_margins[depth])
                        for key, val in closs.items():
                            loss_class[key].append(val)
                    if depth == 0:
                        accuracy['accuracy'].append(acc_avg)
                    for key, val in loss_class.items():
                        loss_event[key].append(sum(val) / len(val))
                for key, val in loss_event.items():
                    loss[key].append(sum(val) / len(val))

        res = {}

        for key, val in loss.items():
            res[key] = sum(val) / len(val)
        res['loss'] += res['distance_estimation']
        res['distance_estimation'] = float(res['distance_estimation'])

        for key, val in accuracy.items():
            res[key] = sum(val) / len(val)

        return res

class FixedCentroidLoss(DiscriminativeLoss):
    '''
    Discriminative Loss without regularization and intercluster loss.
    Here, we use the real-space centroids as the embedding space centroids.
    '''
    def __init__(self, cfg, reduction='sum'):
        super(FixedCentroidLoss, self).__init__(cfg)
        self.loss_config = cfg['modules']['clustering_loss']
        self.num_classes = self.loss_config.get('num_classes', 5)
        self.depth = self.loss_config.get('stride', 5)

        # Clustering Loss Parameters
        self.mseloss = torch.nn.MSELoss(reduction='mean')

        self.dimension = self.loss_config.get('data_dim', 3)
        self.norm = self.loss_config.get('norm', 2)
        self.use_segmentation = self.loss_config.get('use_segmentation', True)


    def get_offsets(self, features, labels, coords):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        loss = 0.0
        cluster_means = self.find_cluster_means(coords, labels)
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            offset = cluster_means[i] - coords[index]
            dists = torch.norm(features[index] - offset,
                               p=1,
                               dim=1)
            l = torch.mean(dists)
            loss += l
        loss /= n_clusters
        return loss

    def combine_multiclass(self, features, slabels, clabels, coords):
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
        loss, acc_segs = [], defaultdict(float)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = (slabels == sc)
            loss.append(self.get_offsets(features[index], clabels[index], coords[index]))
            acc = self.compute_heuristic_accuracy(features[index] + coords[index], clabels[index])
            acc_segs['accuracy_{}'.format(sc.item())] = acc
        return loss, acc_segs


    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc. 
        '''
        num_gpus = len(semantic_labels)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i in range(num_gpus):
            slabels = semantic_labels[i][:, -1]
            coords = semantic_labels[i][:, :3].float()
            if torch.cuda.is_available():
                coords = coords.cuda()
            slabels = slabels.int()
            clabels = group_labels[i][:, -1]
            batch_idx = semantic_labels[i][:, 3]
            embedding = out['cluster_feature'][i]
            nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]
                coords_batch = coords[batch_idx == bidx]
                if self.use_segmentation:
                    loss_class, acc_segs = self.combine_multiclass(
                        embedding_batch, slabels_batch, clabels_batch, coords_batch)
                    loss_class = sum(loss_class) / float(len(loss_class))
                    loss['loss'].append(loss_class)
                    for s, acc in acc_segs.items():
                        accuracy[s].append(acc)
                    acc = sum(acc_segs.values()) / len(acc_segs.values())
                    accuracy['accuracy'].append(acc)
                else:
                    loss["loss"].append(self.combine(embedding_batch, clabels_batch, coords_batch))
                    acc, _ = self.compute_heuristic_accuracy(embedding_batch, clabels_batch)
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

        return res


# TODO:
class SpatialEmbeddingsLoss(nn.Module):

    def __init__(self, cfg, name='clustering_loss'):
        super(SpatialEmbeddingsLoss, self).__init__()
        self.loss_config = cfg['modules'][name]
        self.seediness_weight = self.loss_config.get('seediness_weight', 1.0)
        self.embedding_weight = self.loss_config.get('embedding_weight', 10.0)
        self.smoothing_weight = self.loss_config.get('smoothing_weight', 1.0)
        self.spatial_size = self.loss_config.get('spatial_size', 512)
        
        # Smooth L1 for Embedding Loss
        self.bceloss = torch.nn.BCELoss(reduction='mean')
        # L2 Loss for Seediness and Smoothing
        self.l2loss = torch.nn.MSELoss(reduction='mean')

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

    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(coords, labels)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        acc = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1
            mask[~index] = 0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.sum(torch.pow(embeddings - centroids[i], 2), dim=1)
            p = torch.exp(-dists / (2 * torch.pow(sigma, 2)))
            probs[index] = p[index]
            loss += self.bceloss(p, mask)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))
            # Compute IOU for accuracy
            # with torch.no_grad():
            #     prediction = (p > 0.5).long()
            #     intersection = prediction & mask.long()
            #     union = prediction | mask.long()
            #     acc += float(sum(intersection)) / float(sum(union))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters

        return loss, smoothing_loss, probs, acc


    def combine_multiclass(self, embeddings, margins, seediness, slabels, clabels, coords):
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
            mask_loss, smoothing_loss, probs, acc = self.get_per_class_probabilities(
                embeddings[index], margins[index], clabels[index], coords[index])
            prob_truth = probs.detach()
            seed_loss = self.l2loss(prob_truth, seediness[index].squeeze(1))
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


    def forward(self, out, segment_label, group_label):

        num_gpus = len(segment_label)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        for i in range(num_gpus):
            slabels = segment_label[i][:, -1]
            coords = segment_label[i][:, :3].float()
            if torch.cuda.is_available():
                coords = coords.cuda()
            slabels = slabels.int()
            clabels = group_label[i][:, -1]
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

        print(res)

        return res


class SpatialEmbeddingsLoss2(SpatialEmbeddingsLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(SpatialEmbeddingsLoss2, self).__init__(cfg)

    def get_per_class_probabilities(self, embeddings, margins, labels, coords):
        '''
        Computes binary foreground/background loss.
        '''
        loss = 0.0
        smoothing_loss = 0.0
        centroids = self.find_cluster_means(embeddings, labels)
        n_clusters = len(centroids)
        cluster_labels = labels.unique(sorted=True)
        probs = torch.zeros(embeddings.shape[0]).float().cuda()
        acc = 0.0

        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            mask = torch.zeros(embeddings.shape[0]).cuda()
            mask[index] = 1
            mask[~index] = 0
            sigma = torch.mean(margins[index], dim=0)
            dists = torch.sum(torch.pow(embeddings - centroids[i], 2), dim=1)
            p = torch.exp(-dists / (2 * torch.pow(sigma, 2)))
            probs[index] = p[index]
            loss += self.bceloss(p, mask)
            sigma_detach = sigma.detach()
            smoothing_loss += torch.sum(torch.pow(margins[index] - sigma_detach, 2))
            # Compute IOU for accuracy
            # with torch.no_grad():
            #     prediction = (p > 0.5).long()
            #     intersection = prediction & mask.long()
            #     union = prediction | mask.long()
            #     acc += float(sum(intersection)) / float(sum(union))

        loss /= n_clusters
        smoothing_loss /= n_clusters
        acc /= n_clusters

        return loss, smoothing_loss, probs, acc
    
# Sequential Mask Loss
class SequentialMaskLoss(nn.Module):
    pass

# Focal Loss
class FocalLoss(nn.Module):
    pass

def lovasz_hinge(logits, labels):
    '''
    Binary Lovasz Hinge Loss. Note that the Pytorch implementation of the
    Lovasz Hinge and Softmax Losses are provided in:

    https://github.com/bermanmaxim/LovaszSoftmax

    Again, we modify parts of it to integrate with Sparse Tensors. 

    INPUTS:
        - logits (N x None Tensor): logits for each voxel, (-\infty, \infty)
        - labels (N x None Tensor): binary ground truth labels (0 or 1)

    RETURNS:
        - loss: computed Lovasz-Hinge loss for the current selected N voxels.
    '''
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    # Sort the errors in decreasing order (Lovasz Extension)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
