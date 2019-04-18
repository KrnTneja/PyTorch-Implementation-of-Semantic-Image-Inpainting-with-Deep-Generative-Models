import torch
from torch.utils.data import Dataset 
from skimage import transform
from scipy.io import loadmat
import numpy as np

import glob
import os

import fft
import subsample

class MRIImages(Dataset):
    def __init__(self, directory, image_size=(64,64), transform=None):
        self.directory = directory
        self.images_filename = glob.glob(os.path.join(directory, "*.png"))
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):
        target_image = np.float64(io.imread(self.images_filename[idx]))
        target_image = transform.resize(target_image, self.image_size)
        target_image = (target_image-np.mean(target_image))/np.max(np.abs(target_image))
        return torch.FloatTensor(target_image)

def weighted_mask(mask,window_size):
    assert len(masks.shape) == 2 # 3d input, 0th: batch
    assert window_size % 2 == 1 # odd window size
    max_shift = window_size//2
    output = np.zeros_like(masks)
    for i in range(-max_shift,max_shift+1):
        for j in range(-max_shift,max_shift+1):
            if i != 0 or j != 0:
                outputs += np.roll(masks, (i,j), axis=(0,1))
    outputs = 1 - outputs/(window_size**2-1)
    return outputs*mask

def RandomPatchDataset(Dataset)
    def __init__(self, directory, image_size=(64,64), weighted_mask=True, window_size=7):
        self.directory = directory
        self.images_filename = glob.glob(os.path.join(directory, "*.png"))
        self.image_size = image_size
        self.transform = transform
        self.weighted_mask = weighted_mask
        self.window_size = window_size

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):
        target_image = np.float64(io.imread(self.images_filename[idx]))
        target_image = transform.resize(target_image, self.image_size)
        target_image = (target_image-np.mean(target_image))/np.max(np.abs(target_image))

        # Patch
        mask = np.ones(self.image_size,dtype=np.float32)
        x = np.random.randint(self.image_size[0]//6,5*self.image_size[0]//6)
        y = np.random.randint(self.image_size[1]//6,5*self.image_size[1]//6)
        h = np.random.randint(self.image_size[0]//8,self.image_size[0]//4)
        w = np.random.randint(self.image_size[1]//8,self.image_size[1]//4)
        mask[max(0,x-h//2):min(self.image_size[0],x+h//2),max(0,y-w//2):min(self.image_size[1],y+w//2)] = 0
        target_image = target_image*mask

        # Weighted Mask
        if self.weighted_mask: mask = weighted_mask(mask,self.window_size)

        return torch.FloatTensor(target_image), torch.FloatTensor(mask)
