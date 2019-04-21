import torch
from torch.utils.data import Dataset 

from skimage import transform
import skimage.io as io
import numpy as np

import glob
import os

class GANImages(Dataset):
    def __init__(self, directory, image_size=(96,96)):
        self.directory = directory
        self.images_filename = glob.glob(os.path.join(directory, "*.png"))
        self.image_size = image_size

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):
        target_image = np.float64(io.imread(self.images_filename[idx]))
        target_image = transform.resize(target_image, self.image_size)
        target_image = target_image.reshape((1,)+self.image_size)
        target_image = (target_image-np.mean(target_image))/np.max(np.abs(target_image))
        return torch.FloatTensor(target_image)

def get_weighted_mask(mask,window_size):
    assert len(mask.shape) == 3
    assert window_size % 2 == 1 # odd window size
    max_shift = window_size//2
    output = np.zeros_like(mask)
    for i in range(-max_shift,max_shift+1):
        for j in range(-max_shift,max_shift+1):
            if i != 0 or j != 0:
                output += np.roll(mask, (i,j), axis=(1,2))
    output = 1 - output/(window_size**2-1)
    return output*mask

class RandomPatchDataset(Dataset):
    def __init__(self, directory, image_size=(96,96), weighted_mask=True, window_size=7):
        self.directory = directory
        self.images_filename = glob.glob(os.path.join(directory, "*.png"))
        self.image_size = image_size
        self.weighted_mask = weighted_mask
        self.window_size = window_size

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):
        original_image = np.float64(io.imread(self.images_filename[idx]))
        original_image = transform.resize(original_image, self.image_size)
        original_image = original_image.reshape((1,)+self.image_size)
        original_image = (original_image-np.mean(original_image))/np.max(np.abs(original_image))

        # Patch
        mask = np.ones(self.image_size,dtype=np.float32)
        x = np.random.randint(self.image_size[0]//6,5*self.image_size[0]//6)
        y = np.random.randint(self.image_size[1]//6,5*self.image_size[1]//6)
        h = np.random.randint(self.image_size[0]//4,self.image_size[0]//2)
        w = np.random.randint(self.image_size[1]//4,self.image_size[1]//2)
        mask[max(0,x-h//2):min(self.image_size[0],x+h//2),max(0,y-w//2):min(self.image_size[1],y+w//2)] = 0
        target_image = original_image.copy()
        target_image[0][1-mask > 0.5] = np.max(target_image)

        mask = mask.reshape((1,)+mask.shape)

        # Weighted Mask
        if self.weighted_mask: 
            weighted_mask = get_weighted_mask(mask,self.window_size)
            return torch.FloatTensor(target_image), torch.FloatTensor(original_image), torch.FloatTensor(mask), torch.FloatTensor(weighted_mask)
        else:
            return torch.FloatTensor(target_image), torch.FloatTensor(original_image), torch.FloatTensor(mask)
