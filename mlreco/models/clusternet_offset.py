import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn

from collections import defaultdict
import sklearn.cluster as skc
from sklearn.metrics import adjusted_rand_score

from discriminative_loss import UResNet, DiscriminativeLoss

class OVRNet(nn.Module):
    """
    Clustering by offset vector regression, which is inspired from
    Hugh voting in computer vision.

    Original Paper: https://www.albany.edu/~yl149995/papers/avss2018.pdf
    """

    def __init__(self, cfg, name='clusternet_offset'):
        super(OVRNet, self).__init__()
        import sparseconvnet as scn
        model_config = cfg['modules']['discriminative_loss']

        # UResNet Backbone (Multilayer Loss)
        self.net = UResNet(cfg)

        # Available coordConv modes
        self._coordConv = model_config['coordConv']

        # Configurations for offset regression model.
        self.add_coords = scn.AddTable()
        
    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        #print(input)
        point_cloud, = input
        coords = point_cloud[:, :-2].float()
        features = point_cloud[:, -1][:, None].float()
        if self._coordConv:
            normalized_coords = (coords - self.spatial_size / 2) \
                / float(self.spatial_size / 2)
            # if self._coordConv == 1:
            emb = self.net((coords, features))
            emb = torch.cat([emb, normalized_coords], dim=1)
            # elif self._coordConv == 2:
            #     features = torch.cat([features, normalized_coords], dim=1)
            #     emb = self.net((coords, features))
            # else:
            #     raise ValueError('Invalid CoordConv Layer Configuration: {}'.format(self._coordConv))
        # Output tensor is u + x, where network learn u, the offset vector.
        emb = self.add_coords([emb, coords])

        return {
            'cluster_features': [emb]
        }


class OffsetLoss(nn.Module):

    def __init__(self, cfg, reduction='sum'):
        super(OffsetLoss, self).__init__()
        self._cfg = cfg['modules']['offset_loss']
        self._loss_mode = self._cfg['loss_mode']
        self._hypDim = self._cfg['hypDim']
        self._discriminative_loss = DiscriminativeLoss(cfg)

    def regression_loss(self, features, coords, labels, 
                        cluster_means, margin=1, norm=2):
        """
        Lp Regression Loss for coordinate space offset vector regression. 
        """
        coord_centroids = self._discriminative_loss.find_cluster_means(coords, labels)
        dist_real = torch.mean(torch.norm(features - coords, dim=1, p=norm))
        return [dist_real]

    def combine_multiclass_regression(self, features, slabels, clabels):
        '''
        Wrapper function for combining different components of the loss, 
        in particular when clustering must be done PER SEMANTIC CLASS. 

        INPUTS: 
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class. 
            loss[i] = computed DLoss for semantic class <i>. 
            acc_segs (list): list of computed clustering accuracy for each semantic class. 
        '''
        loss_dict, acc_dict = defaultdict(list), defaultdict(list)
        semantic_classes = slabels.unique()
        for sc in semantic_classes:
            index = slabels == sc
            total_loss = 0.0
            loss_blob = self.regression_loss(features[:, :3][index], clabels[index])
            loss_dict['regression_loss'].append(float(loss_blob[0]))
            total_loss += loss_blob[0]
            loss_hinge = self.combine(features[:, 3:][index], clabels[index])
            total_loss += loss_hinge[0]
            loss_dict['total_loss'].append(total_loss)
            loss_dict['var_loss'].append(loss_hinge[1])
            loss_dict['dist_loss'].append(loss_hinge[2])
            loss_dict['reg_loss'].append(loss_hinge[3])
            acc = self.acc_DUResNet(features[index], clabels[index])
            acc_segs[sc.item()] = acc
        return loss, acc_segs

    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for offset vector + discriminative clustering model.

        INPUTS:
            - out: UResNet output
            - semantic_labels: ground-truth semantic labels
            - group_labels: ground-truth instance labels

        RETURNS:
            - res (dict): A dictionary containing key-value pairs for loss, acc, etc.
        '''
        slabels = semantic_labels[0][:, -1]
        clabels = group_labels[0][:, -1]
        batch_idx = semantic_labels[0][:, 3]
        embedding = out['cluster_features']

        loss = defaultdict(list)
        accuracy = defaultdict(list)
        nbatch = int(batch_idx.unique().shape[0])
        # Loop over each minibatch instance event
        for bidx in batch_idx.unique(sorted=True):
            embedding_batch = embedding[batch_idx == bidx]
            slabels_batch = slabels[batch_idx == bidx]
            clabels_batch = clabels[batch_idx == bidx]

            # Computing the Discriminative Loss
            if self._cfg['multiclass']:
                loss_dict, acc_segs = self.combine_multiclass_regression(
                    embedding_batch, slabels_batch, clabels_batch)
                #print(acc_segs)
                loss["total_loss"].append(
                    sum(loss_dict["total_loss"]) / float(len(loss_dict["total_loss"])))
                loss["var_loss"].append(
                    sum(loss_dict["var_loss"]) / float(len(loss_dict["var_loss"])))
                loss["dist_loss"].append(
                    sum(loss_dict["dist_loss"]) / float(len(loss_dict["dist_loss"])))
                loss["reg_loss"].append(
                    sum(loss_dict["reg_loss"]) / float(len(loss_dict["reg_loss"])))
                loss["regression_loss"].append(
                    sum(loss_dict["regression_loss"]) / float(len(loss_dict["regression_loss"])))
                for s, acc in acc_segs.items():
                    accuracy[s].append(acc)
            else:
                # DEPRECATED
                loss["total_loss"].append(self.combine(embedding_batch, clabels_batch))
                acc, _ = self.acc_DUResNet(embedding_batch, clabels_batch)
                accuracy.append(acc)

        total_loss = sum(loss["total_loss"]) / nbatch
        var_loss = sum(loss["var_loss"]) / nbatch
        dist_loss = sum(loss["dist_loss"]) / nbatch
        reg_loss = sum(loss["reg_loss"]) / nbatch
        regression_loss = sum(loss["regression_loss"]) / nbatch
        acc_segs = defaultdict(float)
        acc_avg = []
        for i in range(5):
            if accuracy[i]:
                acc_segs[i] = sum(accuracy[i]) / float(len(accuracy[i]))
                acc_avg.append(acc_segs[i])
            else:
                acc_segs[i] = 0.0
        acc_avg = sum(acc_avg) / float(len(acc_avg))


        return {
            "loss": total_loss,
            "var_loss": var_loss,
            "dist_loss": dist_loss,
            "reg_loss": reg_loss,
            "regression_loss": regression_loss,
            "accuracy": acc_avg,
            "acc_0": acc_segs[0],
            "acc_1": acc_segs[1],
            "acc_2": acc_segs[2],
            "acc_3": acc_segs[3],
            "acc_4": acc_segs[4]
        }
