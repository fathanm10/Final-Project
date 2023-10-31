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


def compose(size, face_detection, face_detection_crop, pad_ratio, hist, clip_limit, nbins, crop, median, median_size, median_before, unsharp, unsharp_radius, unsharp_amount, unsharp_after, median_unsharp, median_unsharp_size, median_unsharp_amount, median_hsv, median_hsv_method, median_hsv_size, adjust_log, adjust_log_before, flip, autocontrast, random):
    transforms_list = []
    if crop:
        transforms_list += [transforms.CenterCrop(crop)]
    
    if random:
        transforms_list += [
            transforms.RandomHorizontalFlip(.5),
            transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=0),
        ]
        
    if autocontrast:
        transforms_list += [AutoContrast()]
        
    if face_detection:
        transforms_list += [FaceDetectionCrop(pad_ratio=pad_ratio,crop=face_detection_crop)]
        
    if size:
        transforms_list += [transforms.Resize((size, size))]
    
    if median_before:
        transforms_list += [MedianFilter(size=median_size)]
    
    if unsharp:
        transforms_list += [UnsharpFilter(radius=unsharp_radius, amount=unsharp_amount)]
        
    if hist:
        transforms_list += [HistogramEqualization(method=hist, clip_limit=clip_limit,nbins=nbins)]
        
    if median_unsharp:
        transforms_list += [MedianUnsharpFilter(size=median_unsharp_size, amount=median_unsharp_amount)]
        
    if adjust_log_before:
        transforms_list += [AdjustLog()]
        
    if median_hsv:
        transforms_list += [MedianHSVFilter(method=median_hsv_method, size=median_hsv_size)]
        
    if median:
        transforms_list += [MedianFilter(size=median_size)]
        
    if adjust_log:
        transforms_list += [AdjustLog()]
    
    if unsharp_after:
        transforms_list += [UnsharpFilter(radius=unsharp_radius, amount=unsharp_amount)]
    
    transforms_list += [transforms.ToTensor()]

    compose = transforms.Compose(transforms_list)
    return compose


def make_dataset(name, split, max_classes=None, min_samples=None, image_size=100, face_detection=True, face_detection_crop=125, pad_ratio=1, hist=False, clip_limit=.01, nbins=512, crop=False, median=False, median_size=3, median_before=False, unsharp=False, unsharp_radius=20, unsharp_amount=1, unsharp_after=False, median_unsharp=False, median_unsharp_size=3, median_unsharp_amount=1, median_hsv=False, median_hsv_method='sv', median_hsv_size=1, adjust_log=False, adjust_log_before=False, flip=False, autocontrast=False, random=False):
    comp = compose(image_size, face_detection, face_detection_crop, pad_ratio, hist, clip_limit, nbins, crop, median, median_size, median_before, unsharp, unsharp_radius, unsharp_amount, unsharp_after, median_unsharp, median_unsharp_size, median_unsharp_amount, median_hsv, median_hsv_method, median_hsv_size, adjust_log, adjust_log_before, flip, autocontrast, random)
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
