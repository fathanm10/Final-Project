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
kernel_size = 3
padding = 1
num_channels = 3
H,W = [100]*2


class SEModule(nn.Module):
    def __init__(self, embedding_size, ratio=16):
        super(SEModule, self).__init__()
        self.embedding_size = embedding_size
        self.ratio = 16
        
    def forward(self, x):
        h, w, c = x.size()[1:]
        hidden_units = c // ratio
        squeeze = F.avg_pool2d(x, (h, w))
        excitation = squeeze.view(squeeze.size(0), -1)
        fc1 = nn.Linear(512, embedding_size)
        fc2 = nn.Linear(embedding_size, embedding_size)
        self._initialize_weights([fc1, fc2])
        excitation = F.relu(fc1(excitation))
        excitation = fc2(excitation).sigmoid()
        excitation = excitation.view(-1, 1, 1, c)
        return x * excitation
    
    def _initialize_weights(self, layer=None):
        if hasattr(layer, "__getitem__"):
            for l in layer:
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)
        else:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)


class SphereNet(nn.Module):
    def __init__(self, embedding_size=512, model_version='64'):
        super(SphereNet, self).__init__()

        num_layers, num_kernels = model_params[model_version]
        
        self.embedding_size = embedding_size
        self.conv1_shortcut = nn.Conv2d(num_channels, num_kernels[0], kernel_size=kernel_size, padding=padding)
        self.conv1 = self.conv_module(num_layers[0], num_kernels[0])
        self.conv2_shortcut = nn.Conv2d(num_kernels[0], num_kernels[1], kernel_size=kernel_size, padding=padding)
        self.conv2 = self.conv_module(num_layers[1], num_kernels[1])
        self.conv3_shortcut = nn.Conv2d(num_kernels[1], num_kernels[2], kernel_size=kernel_size, padding=padding)
        self.conv3 = self.conv_module(num_layers[2], num_kernels[2])
        self.conv4_shortcut = nn.Conv2d(num_kernels[2], num_kernels[3], kernel_size=kernel_size, padding=padding)
        self.conv4 = self.conv_module(num_layers[3], num_kernels[3])
        self.flatten = nn.Flatten()
        self.fc_bottleneck = nn.Linear(self.embedding_size*H*W, self.embedding_size)
        self._initialize_weights()

    def parametric_relu(self, x):
        alpha = torch.zeros((1, 1, x.size(-1))).to(x.device)
        return F.relu(x) + alpha * torch.min(torch.zeros_like(x), x)

    def se_module(self, x, ratio=16):
        h, w, c = x.size()[1:]
        hidden_units = c // ratio
        squeeze = F.avg_pool2d(x, (h, w))
        excitation = squeeze.view(squeeze.size(0), -1)
        fc1 = nn.Linear(512, embedding_size)
        fc2 = nn.Linear(embedding_size, embedding_size)
        self._initialize_weights([fc1, fc2])
        excitation = F.relu(fc1(excitation))
        excitation = fc2(excitation).sigmoid()
        excitation = excitation.view(-1, 1, 1, c)
        return x * excitation

    def conv_module(self, num_res_layers, num_kernels, use_se=False):
        layers = []
        for _ in range(num_res_layers):
            layers.append(nn.Conv2d(num_kernels, num_kernels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(num_kernels, num_kernels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            if use_se:
                layers.append(SEModule(embedding_size=self.embedding_size))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        def conv_shortcut_loop(x, conv_shortcut, conv):
            net = conv_shortcut(x)
            shortcut = net
            net = conv(net)
            net = net + shortcut
            return net
        
        net = conv_shortcut_loop(x, self.conv1_shortcut, self.conv1)
        net = conv_shortcut_loop(net, self.conv2_shortcut, self.conv2)
        net = conv_shortcut_loop(net, self.conv3_shortcut, self.conv3)
        net = conv_shortcut_loop(net, self.conv4_shortcut, self.conv4)
        net = self.flatten(net)
        conv_final = net
        net = self.fc_bottleneck(net)
        mu = F.normalize(net, dim=1)

        return mu, conv_final
    
    def _initialize_weights(self, layer=None):
        if layer == None:
            layers = [
                self.conv1_shortcut,
                self.conv2_shortcut,
                self.conv3_shortcut,
                self.conv4_shortcut,
                self.fc_bottleneck
            ]
            for l in layers:
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)
        elif hasattr(layer, "__getitem__"):
            for l in layer:
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)
        else:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

