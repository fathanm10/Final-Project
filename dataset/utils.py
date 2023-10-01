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
from collections import Counter
import numpy as np
from .dataset import LFWCustom
from .transform import *
import time

root='./data'


def compose(size, lbp, gabor, hist, crop, custom):
    transforms_list = []
    if crop:
        transforms_list += [transforms.CenterCrop(crop)]
    
    transforms_list += [
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=1 if lbp else 1)  # Convert to Grayscale RGB
    ]
        
    if hist:
        transforms_list += [HistogramEqualizationTransform(method=hist)]
        
    if gabor:
        transforms_list += [GaborFilterTransform(frequency=gabor['frequency'], theta=gabor['theta'])]

    if lbp:
        transforms_list += [LocalBinaryPatternTransform(method=lbp)]
    
    if custom:
        transforms_list += [CustomTransform(custom)]
        
    transforms_list += [transforms.ToTensor()]

    compose = transforms.Compose(transforms_list)
    return compose


def make_dataset(name, split, max_classes=None, min_samples=None, image_size=100, lbp=False, gabor=False, hist=False, crop=False, custom=False):
    comp = compose(image_size, lbp, gabor, hist, crop, custom)
    if name == 'LFW':
        dataset = torchvision.datasets.LFWPeople(
            root=root,
            split=split,
            transform=comp,
            download=True
        )
    if name == 'LFWCustom':
        dataset = LFWCustom(
            root=root,
            split=split,
            max_classes=max_classes,
            min_samples=min_samples,
            transform=comp
        )
    if name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True if split=='train' else False,
            transform=comp,
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
    num_classes = None
    try: num_classes = dataset.nb_classes()
    except:
        try: num_classes = len(dataset.class_to_idx)
        except: num_classes = len(Counter(np.array(dataset, dtype=object)[:,1]))
    return num_classes


def fetch_time(loader):
    then=time.time()
    next(iter(loader))
    return time.time()-then
