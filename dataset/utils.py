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
from tqdm.notebook import tqdm
import shutil
import tarfile
from PIL import Image


def compose(image_size=0,
             face_detection=0,
             face_detection_crop=100,
             pad_ratio=1,
             hist=0,
             clip_limit=.01,
             nbins=512,
             crop=0,
             crop_size=100,
             median=0,
             median_size=3,
             mean=0,
             mean_size=1,
             unsharp=0,
             unsharp_radius=5,
             unsharp_amount=1,
             median_unsharp=0,
             median_unsharp_size=3,
             median_unsharp_amount=1,
             mean_unsharp=0,
             mean_unsharp_size=3,
             mean_unsharp_amount=1,
             autocontrast=0,
             random=0,
             post_process=0,
             margin=0,
             normalize=0,
             norm_mean=0,
             norm_std=0,
             random_resized_crop=100,
             gaussian=0,
             gaussian_sigma=1,
             adjust_log=0,
             adjust_log_gain=1,
            tensor=1
           ):
    transforms_list = []
    
    if crop:
        transforms_list += [transforms.CenterCrop(crop_size)]
    
    if image_size:
        transforms_list += [transforms.Resize((image_size,image_size))]
    
    if face_detection:
        transforms_list += [FaceDetectionCrop(pad_ratio=pad_ratio,crop=face_detection_crop)]
        
    if image_size:
        transforms_list += [transforms.Resize((image_size,image_size))]
    
    if random:
        transforms_list += [
            transforms.RandomResizedCrop(random_resized_crop),
            transforms.RandomHorizontalFlip(.5),
#             transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=0),
#             transforms.RandomVerticalFlip(.5),
#             transforms.RandomRotation(10),
#             Cutout(max_w_size=8,max_h_size=8,p=1),
#             PseudorandomPixelPlacement()
        ]
        
    if autocontrast:
        transforms_list += [AutoContrast()]
        
    if adjust_log:
        transforms_list += [AdjustLog(gain=adjust_log_gain)]
        
    if hist:
        transforms_list += [HistogramEqualization(clip_limit=clip_limit,nbins=nbins)]
        
    if unsharp:
        transforms_list += [UnsharpFilter(radius=unsharp_radius, amount=unsharp_amount)]
        
    if median_unsharp:
        transforms_list += [MedianUnsharpFilter(size=median_unsharp_size, amount=median_unsharp_amount)]
        
    if mean_unsharp:
        transforms_list += [MeanUnsharpFilter(size=mean_unsharp_size, amount=mean_unsharp_amount)]
    
    if gaussian:
        transforms_list += [GaussianFilter(sigma=gaussian_sigma)]
        
    if mean:
        transforms_list += [MeanFilter(size=mean_size)]
        
    if median:
        transforms_list += [MedianFilter(size=median_size)]
    
    if tensor:
        transforms_list += [transforms.ToTensor()]
    
    if normalize:
        transforms_list += [transforms.Normalize(norm_mean,norm_std)]

    compose = transforms.Compose(transforms_list)
    return compose


def make_dataset(name,
                 split,
                 root='./data',
                 max_classes=None,
                 min_samples=None,
                 image_set='deepfunneled',
                 image_size=0,
                 face_detection=0,
                 face_detection_crop=100,
                 pad_ratio=1,
                 hist=0,
                 clip_limit=.01,
                 nbins=512,
                 crop=0,
                 crop_size=100,
                 median=0,
                 median_size=3,
                 mean=0,
                 mean_size=1,
                 unsharp=0,
                 unsharp_radius=5,
                 unsharp_amount=1,
                 median_unsharp=0,
                 median_unsharp_size=3,
                 median_unsharp_amount=1,
                 mean_unsharp=0,
                 mean_unsharp_size=3,
                 mean_unsharp_amount=1,
                 autocontrast=0,
                 random=0,
                 post_process=0,
                 margin=0,
                 normalize=0,
                 norm_mean=0,
                 norm_std=0,
                 random_resized_crop=100,
                 gaussian=0,
                 gaussian_sigma=1,
                 adjust_log=0,
                 adjust_log_gain=1
                ):
    comp = compose(image_size,
                   face_detection,
                   face_detection_crop,
                   pad_ratio,
                   hist,
                   clip_limit,
                   nbins,
                   crop,
                   crop_size,
                   median,
                   median_size,
                   mean,
                   mean_size,
                   unsharp,
                   unsharp_radius,
                   unsharp_amount,
                   median_unsharp,
                   median_unsharp_size,
                   median_unsharp_amount,
                   mean_unsharp,
                   mean_unsharp_size,
                   mean_unsharp_amount,
                   autocontrast,
                   random,
                   post_process,
                   margin,
                   normalize,
                   norm_mean,
                   norm_std,
                   random_resized_crop,
                   gaussian,
                   gaussian_sigma,
                   adjust_log,
                   adjust_log_gain
                  )
    if name == 'LFW':
        dataset = torchvision.datasets.LFWPeople(
            root=root,
            split=split,
            transform=comp,
            download=True,
            image_set=image_set
        )
    if name == 'LFWCustom':
        dataset = LFWCustom(
            root=root,
            split=split,
            max_classes=max_classes,
            min_samples=min_samples,
            transform=comp
        )
    if name == 'LFWPairs':
        dataset = torchvision.datasets.LFWPairs(
            root=root,
            split=split,
            download=1,
            transform=comp,
            image_set=image_set
        )
    return dataset


def load_dataset(dataset,batch_size,shuffle=True, drop_last=False):
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last        
    )
    return loader


def preprocess_dataset(transform_params):
    transform = compose(tensor=0, **transform_params)

    dataset = torchvision.datasets.LFWPeople(
        **{
            'root': './data',
            'split': '10fold',
            'image_set': 'deepfunneled',
            'download': 1
        }
    )
    tar = tarfile.open("./data/lfw-py/lfw-deepfunneled.tgz")
    tar.extractall("./data/lfw-py")
    tar.close()
    if len(transform_params) == 0:
        return
    for image_path in tqdm(dataset.data, desc='Preprocessing Dataset...'):
        image = Image.open(image_path)
        image = transform(image)
        image = ToImage()(image)
        image.save(image_path)


def get_num_classes(dataset):
    num_classes = None
    try: num_classes = dataset.nb_classes()
    except: num_classes = len(dataset.class_to_idx)
    return num_classes


def fetch_time(loader):
    then=time.time()
    next(iter(loader))
    return time.time()-then


def normalize_dataset(datasets):
    dataset = torch.utils.data.ConcatDataset(datasets)
    N_CHANNELS = 3

    full_loader = load_dataset(dataset,batch_size=1,shuffle=True)

    before = time.time()
    mean = torch.zeros(N_CHANNELS)
    std = torch.zeros(N_CHANNELS)
    print('==> Computing mean and std..')
    for inputs, _labels in tqdm(full_loader):
        for i in range(N_CHANNELS):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean, std)

    print("time elapsed: ", time.time()-before)
    return mean, std


# Code taken from https://github.com/Nikronic/CoarseNet/blob/master/utils/preprocess.py
def calc_mean_std(dataset, batch_size, method='strong'):
    """
    Calculate mean and std of a dataset in lazy mode (online)
    On mode strong, batch size will be discarded because we use batch_size=1 to minimize leaps.

    :param dataset: Dataset object corresponding to your dataset
    :param batch_size: higher size, more accurate approximation
    :param method: weak: fast but less accurate, strong: slow but very accurate - recommended = strong
    :return: A tuple of (mean, std) with size of (3,)
    """

    if method == 'weak':
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=0)
        mean = 0.
        std = 0.
        nb_samples = 0.
        for data in loader:
            data = data[0]
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples

        return mean, std

    elif method == 'strong':
        loader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=0)
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)

        for data in loader:
            data = data[0]
            b, c, h, w = data.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(data, dim=[0, 2, 3])
            sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels

        return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
