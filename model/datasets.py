import torch
from torch.utils.data import Dataset 
from skimage import transform
from scipy.io import loadmat
import numpy as np

import glob
import os

import fft
import subsample

train_images = 
test_images = 

class MRIImages(Dataset):
    def __init__(self, directory, image_shape=(91,109), transform=None):
        self.directory = directory
        self.images_filename = train_images 
        self.transform = transform
        self.image_shape = image_shape

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):


class MRIImagesTest(Dataset):
    def __init__(self, directory, image_shape=(91,109,1,1200), transform=None):
        self.directory = directory
        self.images_filename = test_image
        self.transform = transform
        self.image_shape = image_shape

    def __len__(self):
        return len(self.images_filename)

    def __getitem__(self, idx):

