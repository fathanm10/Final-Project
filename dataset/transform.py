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
from skimage.feature import local_binary_pattern
from skimage.filters import gabor


class CompoundLocalBinaryPatternTransform(object):
    def __init__(self, P=8, R=1):
        self.P = P
        self.R = R
        # n, e, s, w, ne, se, sw, nw
        self.x_offset = [0, 1, 0, -1, 1, 1, -1, -1]
        self.y_offset = [1, 0, -1, 0, 1, -1, -1, 1]
    
    def __call__(self, img):
        return self.CLBP(img, P=self.P, R=self.R)
    
    def CLBP(self, img, P, R):
        if isinstance(img, np.ndarray):
            gray_img = img
        else:
            gray_img = np.array(img)

        lbp_image1 = np.zeros_like(gray_img, dtype=np.uint8)
        lbp_image2 = np.zeros_like(gray_img, dtype=np.uint8)
        height, width = gray_img.shape[:2]

        for y in range(R, height - R):
            for x in range(R, width - R):
                center_pixel = gray_img[y, x]
                binary_code = 0
                magnitudes = []
                current_neighbors = []
                for i in range(P):
                    x_neighbor, y_neighbor = self.get_xy_neighbors(i, x, y)
                    neighbor_pixel = gray_img[y_neighbor, x_neighbor]
                    current_neighbors.append(neighbor_pixel)
                    magnitudes.append(abs(int(neighbor_pixel) - int(center_pixel)))
                mean_magnitude = np.array(magnitudes).mean()
                
                for i in range(len(current_neighbors)//2):
                    neighbor_pixel = current_neighbors[i]
                    if neighbor_pixel >= center_pixel:
                        binary_code += 2**((i*2)+1)
                    if magnitudes[i] > mean_magnitude:
                        binary_code += 2**(i*2)
                lbp_image1[y, x] = binary_code
                
                binary_code = 0
                for i in range(len(current_neighbors)//2):
                    neighbor_pixel = current_neighbors[i+len(current_neighbors)//2]
                    if neighbor_pixel >= center_pixel:
                        binary_code += 2**((i*2)+1)
                    if magnitudes[i] > mean_magnitude:
                        binary_code += 2**(i*2)    
                lbp_image2[y, x] = binary_code
                
        return np.stack([lbp_image1, lbp_image2, gray_img],axis=-1)
    
    def get_xy_neighbors(self, i, x, y, angular=False):
        if angular:
            angle = 2 * np.pi * i / self.P
            x_neighbor = round(x + self.R * np.cos(angle))
            y_neighbor = round(y - self.R * np.sin(angle))
        else:
            x_neighbor = x + self.x_offset[i]
            y_neighbor = y + self.y_offset[i]
        return x_neighbor, y_neighbor


class LocalBinaryPatternTransform(object):
    def __init__(self, P=8, R=1, method='ror'):
        self.P = P
        self.R = R
        self.method = method
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        return np.array(local_binary_pattern(img, self.P, self.R, method=self.method), dtype=np.uint8)


class GaborFilterTransform(object):
    def __init__(self, frequency=.6, theta=1):
        self.frequency = frequency
        self.theta = theta
        self.thetas = np.array([0,1/4,1/2,3/4])*np.pi
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        out = np.zeros(img.shape, dtype=np.uint8)
        for theta in self.thetas:
            out = np.maximum(out, gabor(img, frequency=self.frequency, theta=theta)[1])
        return out
#         return np.array(gabor(img, frequency=self.frequency, theta=self.theta)[0], dtype=np.uint8)
