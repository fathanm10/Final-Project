# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# +
class UncertaintyModule(nn.Module):
    def __init__(self, embedding_size):
        super(UncertaintyModule, self).__init__()
        self.fc1   = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.bn1   = nn.BatchNorm1d(embedding_size, affine=True)
        self.relu  = nn.ReLU(embedding_size)
        self.fc2   = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.bn2   = nn.BatchNorm1d(embedding_size, affine=False)
        self.gamma = nn.Parameter(torch.Tensor([1.0]))
        self.beta  = nn.Parameter(torch.Tensor([-7.0]))   # default = -7.0
        
        nn.init.kaiming_normal_(self.fc1)
        nn.init.kaiming_normal_(self.fc2)
        
#         self.embedding_size = embedding_size

#         self.fc1 = nn.Linear(512, embedding_size)
#         self.bn1 = nn.BatchNorm1d(embedding_size)
#         self.relu = nn.ReLU()
        
#         self.fc_log_sigma_sq = nn.Linear(embedding_size, embedding_size)
#         self.bn_log_sigma_sq = nn.BatchNorm1d(embedding_size)
        
        

    def forward(self, inputs):
        x = inputs.view(inputs.size(0), -1)
        x = self.relu(self.bn1(F.linear(x, F.normalize(self.fc1))))
        x = self.bn2(F.linear(x, F.normalize(self.fc2)))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return x
    
#         x = inputs.view(inputs.size(0), -1)

#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         log_sigma_sq = self.fc_log_sigma_sq(x)
#         log_sigma_sq = self.bn_log_sigma_sq(log_sigma_sq)

#         # Share the gamma and beta for all dimensions
#         log_sigma_sq = scale_and_shift(log_sigma_sq, 1e-4, -7.0)

#         # Add epsilon for sigma_sq for numerical stability
#         log_sigma_sq = torch.log(1e-6 + torch.exp(log_sigma_sq))

#         return log_sigma_sq

def scale_and_shift(x, gamma_init=1.0, beta_init=0.0):
    gamma = nn.Parameter(torch.FloatTensor([gamma_init])).to(device)
    beta = nn.Parameter(torch.FloatTensor([beta_init])).to(device)
    
    return gamma * x + beta
