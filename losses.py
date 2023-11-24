import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss

class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MutualLikelihoodScoreLoss(nn.Module):
    def __init__(self, mean=False):
        super(MutualLikelihoodScoreLoss, self).__init__()
        self.mean=mean
        self.min=sys.maxsize
    
    def negMLS(self, mu_X, sigma_sq_X, mean=False):
        if mean:
            XX = torch.mul(mu_X, mu_X).sum(dim=1, keepdim=True)
            YY = torch.mul(mu_X.T, mu_X.T).sum(dim=0, keepdim=True)
            XY = torch.mm(mu_X, mu_X.T)
            mu_diff = XX + YY - 2 * XY
            sig_sum = sigma_sq_X.mean(dim=1, keepdim=True) + sigma_sq_X.T.sum(dim=0, keepdim=True)
            diff    = mu_diff / (1e-8 + sig_sum) + mu_X.size(1) * torch.log(sig_sum)
            return diff
        else:
            mu_diff = mu_X.unsqueeze(1) - mu_X.unsqueeze(0)
            sig_sum = sigma_sq_X.unsqueeze(1) + sigma_sq_X.unsqueeze(0)
            diff    = torch.mul(mu_diff, mu_diff) / (1e-10 + sig_sum) + torch.log(sig_sum)  # BUG
            diff    = diff.sum(dim=2, keepdim=False)
            return diff

    
    def forward(self, mu_X, log_sigma_sq, gty):
        mu_X     = F.normalize(mu_X) # if mu_X was not normalized by l2
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        if gty.device.type == 'cuda':
            non_diag_mask = non_diag_mask.cuda(0)      
        sig_X    = torch.exp(log_sigma_sq)
        loss_mat = self.negMLS(mu_X, sig_X)
        gty_mask = (torch.eq(gty[:, None], gty[None, :])).int()
        pos_mask = (non_diag_mask * gty_mask) > 0
        pos_loss = loss_mat[pos_mask].mean()
#         if torch.isnan(pos_loss):
#             pos_loss = torch.tensor(sys.maxsize, dtype=torch.float32, requires_grad=True).float().to(device)
        return pos_loss
