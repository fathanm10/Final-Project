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
from skimage.filters import rank, median, unsharp_mask
from skimage.exposure import equalize_adapthist, adjust_log
from skimage.morphology import disk
from skimage.color import rgb2hsv, hsv2rgb
from skimage.restoration import denoise_wavelet
from skimage.metrics import peak_signal_noise_ratio as psnr
from face_recognition import face_locations


class HistogramEqualization(object):
    def __init__(self, method='v', clip_limit=.001, disk_size=30, nbins=512):
        self.method = method
        self.clip_limit = clip_limit
        self.disk_size = disk_size
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
    def __call__(self, img):
        locations = face_locations(np.array(img))
        if len(locations) == 0:
            return img
        U,R,D,L = locations[0]
        return img.crop([L,U,R,D])


class MedianFilter(object):
    def __call__(self, img):
        img = np.array(img)
        return median(img)


class MedianHSVFilter(object):
    def __init__(self, method='v'):
        self.method = method
    def __call__(self, img):
        img = np.array(img)
        hsv_img = rgb2hsv(img)
        if 'h' in self.method:
            hsv_img[:,:,0] = median(hsv_img[:,:,0])
        if 's' in self.method:
            hsv_img[:,:,1] = median(hsv_img[:,:,1])
        if 'v' in self.method:
            hsv_img[:,:,2] = median(hsv_img[:,:,2])
        return hsv2rgb(hsv_img)


class MedianUnsharpFilter(object):
    def __init__(self, amount):
        self.amount = amount
    
    def __call__(self, img):
        img = np.array(img)
        hsv_img = rgb2hsv(img)
        unfiltered_img = hsv2rgb(hsv_img)
        hsv_img[:,:,2] = median(hsv_img[:,:,2])
        filtered_img = hsv2rgb(hsv_img)
        return unfiltered_img + self.amount * (unfiltered_img - filtered_img)


class UnsharpFilter(object):
    def __init__(self, radius, amount, hsv=False):
        self.radius = radius
        self.amount = amount
        self.hsv = hsv
    def __call__(self, img):
        img = np.array(img)
        if self.hsv:
            hsv_img = rgb2hsv(img)
            hsv_img[:,:,2] = unsharp_mask(hsv_img[:,:,2], radius=self.radius, amount=self.amount)
            out = hsv2rgb(hsv_img)
        else:
            out = unsharp_mask(img, radius=self.radius, amount=self.amount)
        return out


class DenoiseWavelet(object):
    def __call__(self, img):
        img = np.array(img)
        return denoise_wavelet(img, convert2ycbcr=True, rescale_sigma=True, channel_axis=-1)


class AdjustLog(object):
    def __call__(self, img):
        img = np.array(img)
        return adjust_log(img, 1)
