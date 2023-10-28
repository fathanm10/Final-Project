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
import numpy as np
from skimage.filters import rank, median, unsharp_mask, gaussian
from skimage.exposure import equalize_adapthist, adjust_log
from skimage.morphology import square, disk, cube, ball
from skimage.color import rgb2hsv, hsv2rgb
from skimage.restoration import denoise_wavelet
from skimage.metrics import peak_signal_noise_ratio as psnr
from face_recognition import face_locations
from torchvision.transforms.functional import autocontrast
from torchvision.transforms import ToPILImage, ToTensor, CenterCrop
from random import random


class HistogramEqualization(object):
    def __init__(self, method='v', clip_limit=.001, nbins=512):
        self.method = method
        self.clip_limit = clip_limit
        self.nbins = nbins
    
    def __call__(self, img):
        img = np.array(img)
        if self.method == 'v':
            out = equalize_adapthist(img, clip_limit=self.clip_limit, nbins=self.nbins)
        else:
            hsv_img = rgb2hsv(img)
            if self.method == 'sv':
                hsv_img[:,:,1:] = equalize_adapthist(hsv_img[:,:,1:], clip_limit=self.clip_limit, nbins=self.nbins)
            elif self.method == 's':
                hsv_img[:,:,1] = equalize_adapthist(hsv_img[:,:,1], clip_limit=self.clip_limit, nbins=self.nbins)
            out = hsv2rgb(hsv_img)
        return out


class FaceDetectionCrop(object):
    def pad(self, coordinates):
        U,R,D,L = coordinates
        width = R - L
        height = D - U

        new_width = width * self.pad_ratio
        new_height = height * self.pad_ratio

        new_L = L - (new_width - width) / 2
        new_U = U - (new_height - height) / 2
        new_R = new_L + new_width
        new_D = new_U + new_height
        return new_U, new_R, new_D, new_L
    
    def __init__(self, pad_ratio=1, crop=125):
        self.pad_ratio=pad_ratio
        self.crop=CenterCrop(crop)
    
    def __call__(self, img):
        locations = face_locations(np.array(img))
        if len(locations) == 0:
            return self.crop(img)
        U,R,D,L = self.pad(locations[0])
        return img.crop([L,U,R,D])


class MedianFilter(object):
    def __init__(self, size=3, footprint_type='cube'):
        if footprint_type == 'cube':
            self.footprint = cube(size)
        elif footprint_type == 'ball':
            self.footprint = ball(size)
            
    def __call__(self, img):
        img = np.array(img)
        return median(img, self.footprint)


class MedianHSVFilter(object):
    def __init__(self, method='v', size=3, footprint_type='square'):
        self.method = method
        if footprint_type == 'square':
            self.footprint = square(size)
        elif footprint_type == 'disk':
            self.footprint = disk(size)
    
    def __call__(self, img):
        img = np.array(img)
        hsv_img = rgb2hsv(img)
        if 'h' in self.method:
            hsv_img[:,:,0] = median(hsv_img[:,:,0], self.footprint)
        if 's' in self.method:
            hsv_img[:,:,1] = median(hsv_img[:,:,1], self.footprint)
        if 'v' in self.method:
            hsv_img[:,:,2] = median(hsv_img[:,:,2], self.footprint)
        return hsv2rgb(np.clip(hsv_img,0,1))


class MedianUnsharpFilter(object):
    def __init__(self, amount=1, size=3, footprint_type='square'):
        self.amount = amount
        if footprint_type == 'square':
            self.footprint = square(size)
        elif footprint_type == 'disk':
            self.footprint = disk(size)
    
    def __call__(self, img):
        img = np.array(img)
        unfiltered_img = img
        hsv_img = rgb2hsv(img)
        unfiltered_img = hsv2rgb(hsv_img)
        hsv_img[:,:,2] = median(hsv_img[:,:,2], self.footprint)
        filtered_img = hsv2rgb(hsv_img)
        return np.clip(unfiltered_img + self.amount * (unfiltered_img - filtered_img), 0, 1)


class UnsharpFilter(object):
    def __init__(self, radius=20, amount=1, hsv=False, preserve_range=False):
        self.radius = radius
        self.amount = amount
        self.hsv = hsv
        self.preserve_range = preserve_range
    def __call__(self, img):
        img = np.array(img)
        if self.hsv:
            hsv_img = rgb2hsv(img)
            hsv_img[:,:,2] = unsharp_mask(
                hsv_img[:,:,2],
                radius=self.radius,
                amount=self.amount,
                preserve_range=self.preserve_range
            )
            out = hsv2rgb(hsv_img)
        else:
            out = unsharp_mask(
                img,
                radius=self.radius,
                amount=self.amount,
                preserve_range=self.preserve_range
            )
        return out


class GaussianBlur(object):
    def __init__(self, sigma=1, p=1):
        self.sigma = sigma
        self.p = p
    def __call__(self, img):
        img = np.array(img)
        if random() < self.p:
            return gaussian(img, sigma=self.sigma)
        else:
            return img


class DenoiseWavelet(object):
    def __call__(self, img):
        img = np.array(img)
        return denoise_wavelet(img, convert2ycbcr=True, rescale_sigma=True, channel_axis=-1)


class AdjustLog(object):
    def __call__(self, img):
        img = np.array(img)
        return adjust_log(img, 1)


class AutoContrast(object):
    def __call__(self, img):
        if type(img).__name__!='Image':
            if type(img).__name__=='ndarray' and img.dtype != 'uint8':
                img = ToTensor()(img)
            img = ToPILImage()(img)
        return autocontrast(img)


class RandomNoise(object):
    def __init__(self, amount=.3):
        self.amount=amount
    def __call__(self, img):
        img = np.array(img)
        if img.dtype == 'uint8':
            img = img / 255
        img = img + self.amount*((0.1**0.5)*np.random.randn(250, 250, 3))
        return ToPILImage()(img)
