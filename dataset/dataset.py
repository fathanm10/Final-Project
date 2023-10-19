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
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import numpy as np


class LFWCustom(Dataset):
    def __init__(self, root, split='10fold', max_classes=None, min_samples=None, transform=None):
        self.root = os.path.join(root, 'lfw-py')
        self.data_dir = os.path.join(self.root, 'lfw_funneled')
        self.split = split
        self.test_ratio = 0.33
        self.num_classes = max_classes
        self.min_samples = min_samples
        self.classes = set()
        self.file_paths, self.labels = self.load_data(max_classes)
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle list slicing
            pairs = []
            for i in range(*idx.indices(len(self))):
                img_path = self.file_paths[i]
                label = self.labels[i]
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                pairs.append((image,label))
            return pairs
        else:
            img_path = self.file_paths[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    def __repr__(self):
        return f"""Dataset LFWCustom
    Number of datapoints: {len(self)}
    Root location: {self.root}
    Alignment: funneled
    Split: {self.split}
    Classes (identities): {self.num_classes}
"""
    def nb_classes(self):
        return self.num_classes
    
    def slice_index(self, length):
        ratio = -self.test_ratio
        if self.split == '10fold':
            ratio = 1
        elif self.split == 'train':
            ratio = 1 - self.test_ratio
        return int(length*ratio)
        
    def load_data(self, max_classes):
        file_paths = []
        labels = []
        num_samples = []
        class_names = os.listdir(self.data_dir)
        class_names.sort()
        
        # update num class if max class mismatch
        if max_classes == None or max_classes >= len(class_names):
            self.num_classes = len(class_names)
        
        i = 0
        for _, class_name in enumerate(class_names):
            if len(self.classes) >= self.num_classes:
                break
            class_dir = os.path.join(self.data_dir, class_name)
            
            if os.path.isdir(class_dir):
                image_files = os.listdir(class_dir)
                samples = len(image_files)
                if self.min_samples and samples < self.min_samples:
                    continue
                image_files = [f for f in image_files if f.endswith('.jpg') or f.endswith('.png')]
                slice_index = self.slice_index(len(image_files))
                if self.split == 'train':
                    image_files = image_files[:slice_index]
                elif self.split == 'test':
                    image_files = image_files[slice_index:]
                num_samples.append(len(image_files))
                file_paths.extend([os.path.join(class_dir, fname) for fname in image_files])
                labels.extend([i] * len(image_files))  # same class, labels is the same
                i+=1
                self.classes.add(class_name)
            
        self.num_samples = num_samples
        
        if len(file_paths) == 0:
            print('WARNING: Empty dataset')
            
        return file_paths, labels
