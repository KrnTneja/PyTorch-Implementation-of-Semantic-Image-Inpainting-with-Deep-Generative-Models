import warnings
warnings.filterwarnings("ignore")

import pdb
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as io

import numpy as np

import glob
import os
import argparse

import models
import datasets

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--gan-path", type=str, default="./checkpoints/model.pth")
    parser.add_argument("--train-data-dir", type=str, default="../data/")
    parser.add_argument("--eval-data-dir", type=str, default="../test_images/")
    parser.add_argument("--test-data-dir", type=str, default="../test_images/")
    parser.add_argument("--eval-only", action='store_true', default=False)
    parser.add_argument("--test-only", action='store_true', default=False)

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--optim-steps", type=int, default=1500)
    parser.add_argument("--prior-weight", type=float, default=0.003)
    args = parser.parse_args()
    return args

def context_loss(corrupted_images, generated_images, masks, weighted=True):
    return torch.sum(((corrupted_images-generated_images)**2)*masks)

class RandomVector(nn.Module):
    def __init__(self, batch_size):
        super(RandomVector, self).__init__()
        self.vector = torch.FloatTensor(np.random.normal(0, 1, (args.batch_size,args.latent_dim,)))

    def forward(self): 
        return self.vector

def inpaint(args):
    print("Starting inpainting ...")
    dataset = datasets.RandomPatchDataset(args.test_data_dir,weighted_mask=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Loading trained GAN model
    saved_gan = torch.load(args.gan_path)
    generator = models.Generator(args).cuda()
    discriminator = models.Discriminator(args).cuda()
    generator.load_state_dict(saved_gan["state_dict_G"])
    discriminator.load_state_dict(saved_gan["state_dict_D"])

    i = 1
    for corrupted_images, masks, original_images in dataloader:
        corrupted_images, masks = corrupted_images.cuda(), masks.cuda()
        z_optimum = torch.nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, (args.batch_size,args.latent_dim,))).cuda())
        optimizer_inpaint = optim.Adam([z_optimum])

        for epoch in range(args.optim_steps):
            optimizer_inpaint.zero_grad()
            generated_images = generator(z_optimum)
            discriminator_opinion = discriminator(generated_images)
            c_loss = context_loss(corrupted_images, generated_images, masks)
            prior_loss = torch.sum(-torch.log(discriminator_opinion))
            inpaint_loss = c_loss + args.prior_weight*prior_loss
            inpaint_loss.backward()
            optimizer_inpaint.step()
            print("Epoch: {}/{} Context, Prior, Inpaint Loss = {:.3f} {:.3f} {:.3f}\r".format(epoch, args.optim_steps, c_loss, 
                                                                               prior_loss, inpaint_loss),end="")

        print("")
    
        # save outputs
        for img_id in range(args.batch_size):
            save_image(corrupted_images[img_id], "../outputs/corrupted_{}_{}.png".format(i,img_id), normalize=True)
            save_image(generated_images[img_id], "../outputs/output_{}_{}.png".format(i,img_id), normalize=True)
            save_image(original_images[img_id], "../outputs/original_{}_{}.png".format(i,img_id), normalize=True)

        del z_optimum, optimizer_inpaint
        i += 1

if __name__ == "__main__":
    args = get_arguments()
    inpaint(args)
