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
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# +
compose = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.Grayscale(num_output_channels=3),  # Convert to Grayscale RGB
    transforms.ToTensor(),
])

def make_dataset(name,split):
    root='./data'
    if name == 'LFW':
        dataset = torchvision.datasets.LFWPeople(
            root=root,
            split=split,
            transform=compose,
            download=True
        )
    if name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True if split=='train' else False,
            transform=compose,
            download=True
        )
    return dataset

def load_dataset(dataset,batch_size,shuffle=True):
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader

def get_num_classes(dataset):
    return len(dataset.class_to_idx)
