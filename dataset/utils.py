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


def compose(size, face_detection, hist, clip_limit, nbins, crop, median, median_before, unsharp, unsharp_radius, unsharp_amount, unsharp_after, denoise_wavelet, median_unsharp, median_unsharp_amount):
    transforms_list = []
    if crop:
        transforms_list += [transforms.CenterCrop(crop)]
    
    if face_detection:
        transforms_list += [FaceDetectionCrop()]
        
    if size:
        transforms_list += [transforms.Resize((size, size))]
    
    if denoise_wavelet:
        transforms_list += [DenoiseWavelet()]
        
    if median_before:
        transforms_list += [MedianFilter()]
    
    if unsharp:
        transforms_list += [UnsharpFilter(radius=unsharp_radius, amount=unsharp_amount)]
        
    if median_unsharp:
        transforms_list += [MedianUnsharpFilter(amount=median_unsharp_amount)]
        
    if hist:
        transforms_list += [HistogramEqualization(method=hist, clip_limit=clip_limit,nbins=nbins)]
        
    if median:
        transforms_list += [MedianFilter()]
    
    if unsharp_after:
        transforms_list += [UnsharpFilter(radius=unsharp_radius, amount=unsharp_amount)]
        
    transforms_list += [transforms.ToTensor()]

    compose = transforms.Compose(transforms_list)
    return compose


def make_dataset(name, split, max_classes=None, min_samples=None, image_size=100, face_detection=True, hist=False, clip_limit=.01, nbins=512, crop=False, median=False, median_before=False, unsharp=False, unsharp_radius=20, unsharp_amount=1, unsharp_after=False, denoise_wavelet=False, median_unsharp=False, median_unsharp_amount=1):
    comp = compose(image_size, face_detection, hist, clip_limit, nbins, crop, median, median_before, unsharp, unsharp_radius, unsharp_amount, unsharp_after, denoise_wavelet, median_unsharp, median_unsharp_amount)
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
