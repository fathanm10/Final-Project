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
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
from gfpgan import GFPGANer
from PIL import Image
# import albumentations as A
# from facenet_pytorch import MTCNN

class HistogramEqualization(object):
    def __init__(self, clip_limit=.01, nbins=512):
        self.clip_limit = clip_limit
        self.nbins = nbins
    
    def __call__(self, img):
        img = np.array(img)
        out = equalize_adapthist(img, clip_limit=self.clip_limit, nbins=self.nbins)
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
    
    def __init__(self, pad_ratio=1, crop=0):
        self.pad_ratio=pad_ratio
        self.crop=crop
    
    def closest_to_center(self, faces, img_center):
        distances = [np.linalg.norm(np.array([int((f[1] + f[3]) / 2), int((f[0] + f[2]) / 2)]) - np.array(img_center))
                     for f in faces]
        return faces[np.argmin(distances)]
    
    def __call__(self, img):
        np_img = np.array(img)
        locations = face_locations(np_img)
        if len(locations) == 0:
            if self.crop == 0:
                return img
            else:
                return CenterCrop(self.crop)(img)
        if len(locations) == 1:
            selected_face = locations[0]
        else:
            img_center = np.array([np_img.shape[1] / 2, np_img.shape[0] / 2])
            selected_face = self.closest_to_center(locations, img_center)
        
        U,R,D,L = self.pad(selected_face)
        return img.crop([L,U,R,D])


class MedianFilter(object):
    def __init__(self, size=3, footprint_type='square'):
        if footprint_type == 'square':
            self.footprint = square(size)
        elif footprint_type == 'disk':
            self.footprint = disk(size)
        self.size=size
        self.footprint_type=footprint_type
    
    def __call__(self, img):
        img = np.array(img)
        hsv_img = rgb2hsv(img)
        hsv_img[:,:,2] = median(hsv_img[:,:,2], self.footprint)
        return hsv2rgb(np.clip(hsv_img,0,1))


class MeanFilter(object):
    def __init__(self, size=3, footprint_type='square'):
        if footprint_type == 'square':
            self.footprint = square(size)
        elif footprint_type == 'disk':
            self.footprint = disk(size)
        self.size=size
        self.footprint_type=footprint_type
    
    def __call__(self, img):
        img = np.array(img)
        hsv_img = (rgb2hsv(img)*255).astype(np.uint8)
        hsv_img[:,:,2] = rank.mean(hsv_img[:,:,2], self.footprint)
        hsv_img = hsv_img/255
        return np.clip(hsv2rgb(hsv_img),0,1)


class MedianUnsharpFilter(object):
    def __init__(self, amount=1, size=3, footprint_type='square'):
        self.amount = amount
        if footprint_type == 'square':
            self.footprint = square(size)
        elif footprint_type == 'disk':
            self.footprint = disk(size)
        self.size=size
        self.footprint_type=footprint_type
    
    def __call__(self, img):
        img = np.array(img)
        unfiltered_img = hsv2rgb(rgb2hsv(img))
        filtered_img = MedianFilter(size=self.size,footprint_type=self.footprint_type)(img)
        return np.clip(unfiltered_img + self.amount * (unfiltered_img - filtered_img), 0, 1)


class MeanUnsharpFilter(object):
    def __init__(self, amount=1, size=3, footprint_type='square'):
        self.amount = amount
        if footprint_type == 'square':
            self.footprint = square(size)
        elif footprint_type == 'disk':
            self.footprint = disk(size)
        self.size=size
        self.footprint_type=footprint_type
    
    def __call__(self, img):
        img = np.array(img)
        unfiltered_img = hsv2rgb(rgb2hsv(img))
        filtered_img = MeanFilter(size=self.size,footprint_type=self.footprint_type)(img)
        return np.clip(unfiltered_img + self.amount * (unfiltered_img - filtered_img), 0, 1)


class UnsharpFilter(object):
    def __init__(self, radius=1, amount=1, preserve_range=False):
        self.radius = radius
        self.amount = amount
        self.preserve_range = preserve_range
    def __call__(self, img):
        img = np.array(img)
        hsv_img = rgb2hsv(img)
        hsv_img[:,:,2] = unsharp_mask(
            hsv_img[:,:,2],
            radius=self.radius,
            amount=self.amount,
            preserve_range=self.preserve_range
        )
        out = hsv2rgb(hsv_img)
        return out


class GaussianFilter(object):
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
    def __init__(self, gain=1):
        self.gain=gain
    def __call__(self, img):
        img = np.array(img)
        hsv_img = rgb2hsv(img)
        hsv_img[:,:,2] = adjust_log(hsv_img[:,:,2], self.gain)
        out = hsv2rgb(hsv_img)
        return out


class ToImage(object):
    def __call__(self,img):
        if type(img).__name__!='Image':
            if type(img).__name__=='ndarray' and img.dtype != 'uint8':
                img = ToTensor()(img)
            img = ToPILImage()(img)
        return img


class AutoContrast(object):
    def __call__(self, img):
        img = ToImage()(img)
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


# +
# class Cutout(object):
#     def __init__(self, num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=1):
#         self.transform=A.Cutout(num_holes,max_h_size,max_w_size,fill_value,p=p)
    
#     def __call__(self, img):
#         original_class = type(img).__name__
#         img = np.array(img)
#         cutout_img = self.transform(image=img)['image']
#         if original_class == 'Image':
#             cutout_img = ToPILImage()(cutout_img)
#         return cutout_img
# -

class PseudorandomPixelPlacement(object):
    def __init__(self, p=1):
        self.p=p
    
    def __call__(self, img):        
        if random() < self.p:
            original_class = type(img).__name__
            img = np.array(img)
            original_height, original_width, num_channels = img.shape

            # Randomly select dx and dy for each pixel
            dx = np.random.randint(2, size=(original_height // 2, original_width // 2))
            dy = np.random.randint(2, size=(original_height // 2, original_width // 2))

            # Initialize the down-sampled image
            downsampled_height, downsampled_width = original_height // 2, original_width // 2
            downsampled_image = np.zeros((downsampled_height, downsampled_width, num_channels), dtype=img.dtype)

            for x in range(downsampled_height):
                for y in range(downsampled_width):
                    downsampled_image[x, y, :] = img[2 * x + dx[x, y], 2 * y + dy[x, y], :]
            if original_class == 'Image':
                downsampled_image = ToPILImage()(downsampled_image)
            return downsampled_image
        else:
            return img


# +
# class MTCNNFaceDetection(object):
#     def __init__(self, post_process=False, margin=0, crop_size=125):
#         self.mtcnn = MTCNN(device='cuda',post_process=post_process,margin=margin)
#         self.post_process = post_process
#         self.crop = CenterCrop(crop_size)
#     def __call__(self, input_img):
#         img = self.mtcnn(input_img)
#         if img==None:
#             return self.crop(input_img)
#         img = img.permute(1,2,0).numpy()
#         if ~self.post_process:
#             img = img.astype(np.uint8)
#         return ToPILImage()(img)
# -

class GFPGANRestorer(object):
    def __init__(self, upscale=2, bg_upsampler='realesrgan'):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True)  # need to set False in CPU mode
        self.restorer = GFPGANer(
            model_path='./experiments/pretrained_models/GFPGANv1.3.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=bg_upsampler)
    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _,_, restored_img = self.restorer.enhance(img)
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        restored_img = Image.fromarray(restored_img)
        return restored_img
    
