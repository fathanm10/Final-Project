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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# +
class UncertaintyModule(nn.Module):
    def __init__(self, embedding_size):
        super(UncertaintyModule, self).__init__()
        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(512, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU()
        
        self.fc_log_sigma_sq = nn.Linear(embedding_size, embedding_size)
        self.bn_log_sigma_sq = nn.BatchNorm1d(embedding_size)

    def forward(self, inputs, phase_train=True):
        x = inputs.view(inputs.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        log_sigma_sq = self.fc_log_sigma_sq(x)
        log_sigma_sq = self.bn_log_sigma_sq(log_sigma_sq)

        # Share the gamma and beta for all dimensions
        log_sigma_sq = scale_and_shift(log_sigma_sq, 1e-4, -7.0)

        # Add epsilon for sigma_sq for numerical stability
        log_sigma_sq = torch.log(1e-6 + torch.exp(log_sigma_sq))

        return log_sigma_sq

def scale_and_shift(x, gamma_init=1.0, beta_init=0.0):
    gamma = nn.Parameter(torch.FloatTensor([gamma_init])).to(device)
    beta = nn.Parameter(torch.FloatTensor([beta_init])).to(device)
    
    return gamma * x + beta
