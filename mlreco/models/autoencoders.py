import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn

from mlreco.models.layers.autoencoder import ConvAutoEncoder

class AutoEncoder(nn.Module):

    def __init__(self, cfg, name='autoencoder'):
        super(AutoEncoder, self).__init__()
        self.net = ConvAutoEncoder(cfg, name='conv_autoenc')

    def forward(self, input):
        res = self.net(input)
        return res


class AutoEncoderLoss(nn.Module):

    def __init__(self, cfg, name='ae_loss'):
        super(AutoEncoderLoss, self).__init__()
        self.l2loss = nn.MSELoss()

    def forward(self, res, input_data, cluster_label):
        reconstruction = res['reco'][0]
        target = input_data[0][:, -1].view(-1, 1).float()
        clabel = cluster_label[0][:, -2]
        batch_index = cluster_label[0][:, 3]
        loss = []
        for i, bidx in enumerate(batch_index.unique()):
            batch_mask = batch_index == bidx
            reco_batch = reconstruction[batch_mask]
            target_batch = target[batch_mask]
            clabels_batch = clabel[batch_mask]
            batch_loss = []
            for j, cidx in enumerate(clabels_batch.unique()):
                cluster_mask = clabels_batch == cidx
                reco_cluster = reco_batch[cluster_mask]
                target_cluster = target_batch[cluster_mask]
                l = self.l2loss(reco_cluster, target_cluster)
                batch_loss.append(l)
            batch_loss = sum(batch_loss) / len(batch_loss)
            loss.append(batch_loss)
        loss = sum(loss) / len(loss)
        return {'loss': loss, 'accuracy': 0}
