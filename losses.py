import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses

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

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
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


# +
def negative_MLS(X, Y, sigma_sq_X, sigma_sq_Y, mean=False):
    D = X.size(1)
    if mean:
        Y = Y.t()
        XX = torch.sum(X**2, dim=1, keepdim=True)
        YY = torch.sum(Y**2, dim=0, keepdim=True)
        XY = torch.matmul(X, Y)
        diffs = XX + YY - 2*XY

        sigma_sq_Y = sigma_sq_Y.t()
        sigma_sq_X = torch.mean(sigma_sq_X, dim=1, keepdim=True)
        sigma_sq_Y = torch.mean(sigma_sq_Y, dim=0, keepdim=True)
        sigma_sq_fuse = sigma_sq_X + sigma_sq_Y

        diffs = diffs / (1e-8 + sigma_sq_fuse) + D * torch.log(sigma_sq_fuse)

        return diffs
    else:
        X = X.view(-1, 1, D)
        Y = Y.view(1, -1, D)
        sigma_sq_X = sigma_sq_X.view(-1, 1, D)
        sigma_sq_Y = sigma_sq_Y.view(1, -1, D)
        sigma_sq_fuse = sigma_sq_X + sigma_sq_Y
        diffs = (X - Y)**2 / (1e-10 + sigma_sq_fuse) + torch.log(sigma_sq_fuse)
        return torch.sum(diffs, dim=2)

# Use: loss_func(mu, log_sigma_sq, labels)
class MutualLikelihoodScoreLoss(nn.Module):
    def __init__(self):
        super(MutualLikelihoodScoreLoss, self).__init__()
        
    def forward(self, mu, log_sigma_sq, labels):
        batch_size = mu.size(0)
        diag_mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        non_diag_mask = ~diag_mask

        sigma_sq = torch.exp(log_sigma_sq)
        loss_mat = negative_MLS(mu, mu, sigma_sq, sigma_sq)

        label_mat = labels.view(-1, 1) == labels.view(1, -1)
        label_mask_pos = non_diag_mask & label_mat

        loss_pos = loss_mat.masked_select(label_mask_pos)

        return torch.mean(loss_pos)
