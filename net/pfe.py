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
from .uncertainty_module import UncertaintyModule
from .resnet import Resnet50


class PFE(nn.Module):
    def __init__(self, embedding_size, backbone):
        super(PFE, self).__init__()
        self.backbone = backbone
        self.uncertainty_module = UncertaintyModule(embedding_size)

    def forward(self, x):
        mu, conv_final = self.backbone(x)
        log_sigma_sq = self.uncertainty_module(mu)
        return mu, log_sigma_sq
