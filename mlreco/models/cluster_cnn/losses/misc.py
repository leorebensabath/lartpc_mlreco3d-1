import torch
import torch.nn as nn
import torch.nn.functional as F

from .lovasz import StableBCELoss

# Collection of Miscellaneous Loss Functions not yet implemented in Pytorch.

class FocalLoss(nn.Module):
    '''
    Original Paper: https://arxiv.org/abs/1708.02002
    Implementation: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    '''
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.stable_bce = StableBCELoss()

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = self.stable_bce(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class WeightedFocalLoss(FocalLoss):

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(WeightedFocalLoss, self).__init__(alpha=alpha, gamma=gamma, logits=logits, reduce=reduce)

    def forward(self, inputs, targets):
        with torch.no_grad():
            pos_weight = torch.sum(targets == 0) / (1.0 + torch.sum(targets == 1))
            weight = torch.ones(inputs.shape[0]).cuda()
            weight[targets == 1] = pos_weight
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = self.stable_bce(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        F_loss = torch.mul(F_loss, weight)

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class SoftDiceLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super(SoftDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        '''
        inputs: (N x 1) probability float tensor
        targets: (N x 1) label float tensor
        '''
        num = 2 * torch.sum(inputs * targets) + self.eps
        denom = torch.sum(inputs) + torch.sum(targets) + self.eps
        return 1 - num / denom


# class GeneralizedDiceLoss(nn.Module):
#
#     def __init__(self, eps=1e-8):
#         super(GeneralizedDiceLoss, self).__init__()
#         self.eps = eps
#
#     def forward(self, inputs, targets):
#         assert targets.shape[1] == 1
#         gt = F.one_hot(targets)
#         w = 1 / (torch.pow(gt.sum(dim=0), 2) + self.eps)
#         num = 2 * torch.sum(inputs * targets, dim)


class DynamicCrossEntropy(nn.Module):

    def __init__(self, eps=1e-8):
        super(DynamicCrossEntropy, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        '''
        inputs: (N x C) float tensor.
        targets: (N x 1) label float tensor, with labels 0, 1, ..., C.
        '''
        assert targets.shape[1] == 1
        gt = targets.squeeze(1)
        p = (inputs + self.eps) / (torch.sum(inputs, dim=1, keepdim=True) + self.eps)
        loss = -torch.mean(torch.log(torch.sum(p * gt, dim=1)))
        return loss


class SoftF1Loss(nn.Module):

    def __init__(self, beta=0.5, eps=1e-6):
        super(SoftF1Loss, self).__init__()
        # Higher the beta, the more weight to purity.
        assert 0 < beta < 1
        self.beta = beta
        self.eps = eps

    def forward(self, inputs, targets):
        purity = (torch.sum(inputs * targets) + self.eps) \
            / (torch.sum(inputs) + self.eps)
        efficiency = (torch.sum(inputs * targets + self.eps)) \
            / (torch.sum(targets) + self.eps)
        f1 = (purity * efficiency) / (self.beta * efficiency + (1 - self.beta) * purity)
        return 1 - f1
