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

model_params = {
    '4': ([0, 0, 0, 0], [64, 128, 256, 512]),
    '10': ([0, 1, 2, 0], [64, 128, 256, 512]),
    '20': ([1, 2, 4, 1], [64, 128, 256, 512]),
    '36': ([2, 4, 8, 2], [64, 128, 256, 512]),
    '64': ([3, 8, 16, 3], [64, 128, 256, 512]),
}


class SphereNet(nn.Module):
    def __init__(self, embedding_size=512, model_version='64'):
        super(SphereNet, self).__init__()

        num_layers, num_kernels = model_params[model_version]

        self.conv1 = self.conv_module(num_layers[0], num_kernels[0])
        self.conv2 = self.conv_module(num_layers[1], num_kernels[1])
        self.conv3 = self.conv_module(num_layers[2], num_kernels[2])
        self.conv4 = self.conv_module(num_layers[3], num_kernels[3])

        self.fc_bottleneck = self.bottleneck(embedding_size)

    def parametric_relu(self, x):
        alpha = torch.zeros((1, 1, x.size(-1))).to(x.device)
        return F.relu(x) + alpha * torch.min(torch.zeros_like(x), x)

    def se_module(self, x):
        h, w, c = x.size()[1:]
        hidden_units = c // 16  # Assuming ratio=16
        squeeze = F.avg_pool2d(x, (h, w))
        excitation = squeeze.view(squeeze.size(0), -1)
        excitation = F.relu(self.fc1(excitation))
        excitation = self.fc2(excitation).sigmoid()
        excitation = excitation.view(-1, 1, 1, c)
        return x * excitation

    def conv_module(self, num_res_layers, num_kernels, use_se=False):
        layers = []
        for _ in range(num_res_layers):
            layers.append(nn.Conv2d(num_kernels, num_kernels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(num_kernels, num_kernels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if use_se:
                layers.append(self.se_module())
        return nn.Sequential(*layers)

    def bottleneck(self, embedding_size):
        return nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.fc_bottleneck(net)
        conv_final = net
        mu = F.normalize(net, dim=1)

        return mu, conv_final

