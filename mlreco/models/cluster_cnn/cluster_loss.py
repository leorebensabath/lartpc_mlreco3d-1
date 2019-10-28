import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from collections import defaultdict
from mlreco.utils.utils import ForwardData


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
            var_loss: (float) variance loss (see paper).
        '''
        var_loss = 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            index = (labels == c)
            dists = torch.norm(features[index] - cluster_means[i] + 1e-8,
                               p=self.norm,
                               dim=1)
            hinge = torch.clamp(dists - margin, min=0)
            l = torch.mean(torch.pow(hinge, 2))
            var_loss += l
        var_loss /= n_clusters
        return var_loss

    def inter_cluster_loss(self, cluster_means, margin=1.5):
        '''
        Implementation of distance loss in Discriminative Loss.
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): the magnitude of the margin delta_d in the paper.
            Think of it as the distance between each separate clusters in
            embedding space.
        Returns:
            dist_loss (float): computed cross-centroid distance loss (see paper).
            Factor of 2 is included for proper normalization.
        '''
        dist_loss = 0.0
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
                        dist_loss += torch.pow(hinge, 2)
            dist_loss /= float((n_clusters - 1) * n_clusters)
            return dist_loss

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
        delta_var = kwargs.get('intra_margin', 0.5)
        delta_dist = kwargs.get('inter_margin', 1.5)
        intra_weight = kwargs.get('intra_weight', 1.0)
        inter_weight = kwargs.get('inter_weight', 1.0)
        reg_weight = kwargs.get('reg_weight', 0.001)

        c_means = self.find_cluster_means(features, labels)
        inter_loss = self.inter_cluster_loss(c_means, margin=delta_var)
        intra_loss = self.intra_cluster_loss(features,
                                           labels,
                                           c_means,
                                           margin=delta_dist)
        reg_loss = self.regularization(c_means)

        loss = intra_weight * intra_loss + inter_weight \
            * inter_loss + reg_weight * reg_loss

        return {
            'total_loss': loss, 
            'var_loss': intra_weight * float(intra_loss),
            'dist_loss': inter_weight * float(inter_loss),
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
            loss['total_loss'].append(loss_blob['total_loss'])
            loss['var_loss'].append(loss_blob['var_loss'])
            loss['dist_loss'].append(loss_blob['dist_loss'])
            loss['reg_loss'].append(loss_blob['reg_loss'])
            acc = self.compute_heuristic_accuracy(features[index], clabels[index])
            acc_segs[sc.item()] = acc
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
            slabels = semantic_labels[i][:, 4]
            slabels = slabels.type(torch.LongTensor)
            clabels = group_labels[i][:, 4]
            batch_idx = semantic_labels[i][:, 3]
            embedding = out['cluster_feature'][i]
            nbatch = int(batch_idx.unique().shape[0])

            for bidx in batch_idx.unique(sorted=True):
                embedding_batch = embedding[batch_idx == bidx]
                slabels_batch = slabels[batch_idx == bidx]
                clabels_batch = clabels[batch_idx == bidx]

                if self.use_segmentation:
                    loss_dict, acc_segs = self.combine_multiclass(
                        embedding_batch, slabels_batch, clabels_batch, **self.loss_hyperparams)
                    loss["total_loss"].append(
                        sum(loss_dict["total_loss"]) / float(len(loss_dict["total_loss"])))
                    loss["var_loss"].append(
                        sum(loss_dict["var_loss"]) / float(len(loss_dict["var_loss"])))
                    loss["dist_loss"].append(
                        sum(loss_dict["dist_loss"]) / float(len(loss_dict["dist_loss"])))
                    loss["reg_loss"].append(
                        sum(loss_dict["reg_loss"]) / float(len(loss_dict["reg_loss"])))
                    for s, acc in acc_segs.items():
                        accuracy[s].append(acc)
                else:
                    loss["total_loss"].append(self.combine(embedding_batch, clabels_batch, **self.loss_hyperparams))
                    acc, _ = self.compute_heuristic_accuracy(embedding_batch, clabels_batch)
                    accuracy.append(acc)

        total_loss = sum(loss["total_loss"]) / (nbatch * num_gpus)
        var_loss = sum(loss["var_loss"]) / (nbatch * num_gpus)
        dist_loss = sum(loss["dist_loss"]) / (nbatch * num_gpus)
        reg_loss = sum(loss["reg_loss"]) / (nbatch * num_gpus)
        acc_segs = defaultdict(float)
        acc_avg = []
        for i in range(self.num_classes):
            if accuracy[i]:
                acc_segs[i] = sum(accuracy[i]) / float(len(accuracy[i]))
                acc_avg.append(acc_segs[i])
            else:
                acc_segs[i] = 0.0
        acc_avg = sum(acc_avg) / float(len(acc_avg))

        res = {
            "loss": total_loss,
            "var_loss": var_loss,
            "dist_loss": dist_loss,
            "reg_loss": reg_loss,
            "accuracy": acc_avg,
            "acc_0": acc_segs[0],
            "acc_1": acc_segs[1],
            "acc_2": acc_segs[2],
            "acc_3": acc_segs[3],
            "acc_4": acc_segs[4]
        }

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
                loss["total_loss"].append(
                    sum(loss_dict["total_loss"]) / float(len(loss_dict["total_loss"])))
                loss["var_loss"].append(
                    sum(loss_dict["var_loss"]) / float(len(loss_dict["var_loss"])))
                loss["dist_loss"].append(
                    sum(loss_dict["dist_loss"]) / float(len(loss_dict["dist_loss"])))
                loss["reg_loss"].append(
                    sum(loss_dict["reg_loss"]) / float(len(loss_dict["reg_loss"])))
                for s, acc in acc_segs.items():
                    accuracy[s].append(acc)
            else:
                loss["total_loss"].append(self.combine(embedding_batch, clabels_batch, **kwargs))
                acc = self.compute_heuristic_accuracy(embedding_batch, clabels_batch)
                accuracy.append(acc)

        summed_loss = { key : sum(l) for key, l in loss.items() }
        averaged_acc = { key : sum(l) / float(len(l)) for key, l in accuracy.items() }
        return summed_loss, averaged_acc


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
        accuracy = defaultdict(float)

        for i_gpu in range(len(semantic_labels)):
            batch_idx = semantic_labels[i_gpu][0][:, 3].detach().cpu().int().numpy()
            batch_idx = np.unique(batch_idx)
            batch_size = len(batch_idx)
            # Summing clustering loss over layers.
            for i, em in enumerate(out['cluster_feature'][i_gpu]):
                delta_var, delta_dist = self.intra_margins[i], self.inter_margins[i]
                loss_i, acc_i = self.compute_loss_layer(em, semantic_labels[i_gpu][i], group_labels[i_gpu][i], batch_idx,
                                                        delta_var=delta_var, delta_dist=delta_dist)
                for key, val in loss_i.items():
                    loss[key].append(val)
                # Compute accuracy at last layer.
                if i == 0:
                    acc_clustering = acc_i
            for key, acc in acc_clustering.items():
                accuracy[key] = float(acc) * len(batch_idx)

        total_loss = sum(loss["total_loss"])
        var_loss = sum(loss["var_loss"])
        dist_loss = sum(loss["dist_loss"])
        reg_loss = sum(loss["reg_loss"])

        total_acc = 0
        for acc in accuracy.values():
            total_acc += acc / len(accuracy.keys())
        accuracy['accuracy'] = total_acc

        res = {
            "loss": total_loss / batch_size,
            "var_loss": var_loss / batch_size,
            "reg_loss": reg_loss / batch_size,
            "dist_loss": dist_loss / batch_size,
            "acc_0": accuracy[0],
            "acc_1": accuracy[1],
            "acc_2": accuracy[2],
            "acc_3": accuracy[3],
            "acc_4": accuracy[4],
            "accuracy": accuracy['accuracy'] / batch_size
        }

        return res


class AllyEnemyLoss(MultiScaleLoss):

    def __init__(self, cfg):
        super(AllyEnemyLoss, self).__init__(cfg)
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


    def similarity_matrix(self, points):
        """
        Uses BLAS/LAPACK operations to efficiently compute pairwise distances.
        """
        M = points[None,...]
        zeros = torch.zeros(1, 1, 1)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        inner_prod = torch.baddbmm(zeros, M, M.permute([0, 2, 1]), alpha=-2.0, beta=0.0)
        squared    = torch.sum(torch.mul(M, M), dim=-1, keepdim=True)
        inner_prod += squared
        inner_prod += squared.permute([0, 2, 1])
        return inner_prod


    def compute_affinity_loss(self, embedding_class, cluster_class,
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
        dist = self.similarity_matrix(embedding_class).squeeze(0)
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
                        dloss, dlossF, dlossE = self.compute_affinity_loss(
                            embedding_class, cluster_class,
                            ally_margin=self.ally_margins[depth],
                            enemy_margin=self.enemy_margins[depth])
                        # Informations to be saved in log file (loss/accuracy). 
                        data.update_mean('accuracy', acc)
                        data.update_mean('intra_loss', closs['var_loss'])
                        data.update_mean('inter_loss', closs['dist_loss'])
                        data.update_mean('reg_loss', closs['reg_loss'])
                        data.update_mean('ally_loss', dlossF)
                        data.update_mean('enemy_loss', dlossE)
                        data.update_mean('accuracy_{}'.format(class_), acc)
                        data.update_mean('intra_loss_{}'.format(class_), closs['var_loss'])
                        data.update_mean('inter_loss_{}'.format(class_), closs['dist_loss'])
                        data.update_mean('ally_loss_{}'.format(class_), dlossF)
                        data.update_mean('enemy_loss_{}'.format(class_), dlossE)
                        clustering_loss += self.clustering_weight * closs['total_loss']
                        affinity_loss += self.affinity_weight * dloss
                        count += 1

        res = data.as_dict()
        res['loss'] = (clustering_loss + affinity_loss) / count
        print(res)
        return res


class EnhancedDiscLoss(DiscriminativeLoss):

    def __init__(self, cfg, name='clustering_loss'):
        super(EnhancedDiscLoss, self).__init__(cfg)


class StackNetLoss(nn.Module):

    def __init__(self, cfg, name='stacknet_loss'):
        self.loss_config = cfg['modules'][name]
        self.multiScale = self.loss_config.get('multi_scale', False)
        self.loss_funcs = {}
        self.loss_funcs['single'] = DiscriminativeLoss(cfg)
        if self.multi_scale:
            self.loss_funcs['multi'] = MultiScaleLoss(cfg)
        self.final_layer_weight = self.loss_config.get('final_layer_weight', 1.0)
        

    def forward(self, result, segment_label, cluster_label):

        res_final = self.loss_funcs['single'](result)


# TODO:
# Sequential Mask Loss
class SequentialMaskLoss(nn.Module):
    pass

# Lovasz Hinge Loss
class LovaszHinge(nn.Module):
    pass

# Lovasz Softmax Loss
class LovaszSoftmax(nn.Module):
    pass

# Focal Loss
class FocalLoss(nn.Module):
    pass