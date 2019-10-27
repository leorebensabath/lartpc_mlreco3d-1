import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from collections import defaultdict

from mlreco.models.discriminative_loss import UResNet, DiscriminativeLoss

class OVRNet(nn.Module):
    """
    Clustering by offset vector regression, which is inspired from
    Hough voting in computer vision.

    Original Paper: https://www.albany.edu/~yl149995/papers/avss2018.pdf
    """

    def __init__(self, cfg, name='clusternet_offset'):
        super(OVRNet, self).__init__()
        model_config = cfg['modules']['discriminative_loss']

        # UResNet Backbone (Multilayer Loss)
        self.net = UResNet(cfg)

        # Configurations for offset regression model.
        
    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        shape of point_cloud[0] = (N, 4)
        """
        point_cloud, = input
        coords = point_cloud[:, :-2].float()
        net_output = self.net(input)
        embedding = net_output['cluster_feature'][0]
        embedding[:, :3] += coords

        return {
            'cluster_features': [embedding]
        }


class OffsetLoss(nn.Module):

    def __init__(self, cfg, reduction='sum'):
        super(OffsetLoss, self).__init__()
        self._cfg = cfg['modules']['offset_loss']
        self._hypDim = self._cfg.get('hypDim', 8)
        self._regression_loss_weight = self._cfg.get('regression_loss_weight', 1.0)
        self._regression_margin = self._cfg.get('regression_margin', 0.0)
        self._discriminative_loss = DiscriminativeLoss(cfg)

    def regression_loss(self, features, labels, coords, margin=0.0, norm=2):
        """
        Lp Regression Loss for coordinate space offset vector regression. 
        """
        coord_centroids = self._discriminative_loss.find_cluster_means(coords, labels)
        dist_real = torch.norm(features - coords, dim=1, p=norm)
        hinge = torch.clamp(dist_real - margin, min=0)
        l = torch.mean(torch.pow(hinge, 2))
        return l

    def combine_multiclass_regression(self, features, slabels, clabels, coords):
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
            regression_loss = self.regression_loss(features[:, :3][index],
                clabels[index], coords[index])
            loss_dict['regression_loss'].append(float(regression_loss))
            total_loss += self._regression_loss_weight * regression_loss
            loss_hinge = self._discriminative_loss.combine(
                features[:, 3:][index], clabels[index])
            total_loss += loss_hinge[0]
            loss_dict['total_loss'].append(total_loss)
            loss_dict['var_loss'].append(loss_hinge[1])
            loss_dict['dist_loss'].append(loss_hinge[2])
            loss_dict['reg_loss'].append(loss_hinge[3])
            acc = self._discriminative_loss.acc_DUResNet(features[index], clabels[index])
            acc_dict[sc.item()] = acc
        return loss_dict, acc_dict

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
        embedding = out['cluster_features'][0]
        coords = semantic_labels[0][:, :3].float()

        loss = defaultdict(list)
        accuracy = defaultdict(list)
        nbatch = int(batch_idx.unique().shape[0])
        # Loop over each minibatch instance event
        for bidx in batch_idx.unique(sorted=True):
            embedding_batch = embedding[batch_idx == bidx]
            slabels_batch = slabels[batch_idx == bidx]
            clabels_batch = clabels[batch_idx == bidx]
            coords_batch = coords[batch_idx == bidx]

            # Computing the Discriminative Loss
            loss_dict, acc_segs = self.combine_multiclass_regression(
                embedding_batch, slabels_batch, clabels_batch, coords_batch)
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
