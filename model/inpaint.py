import torch as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as io

import numpy as np
import tqdm

import glob
import os
import argparse

import models
import datasets

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gan-path", type=str, default="./model.pth")
    parser.add_argument("--new-model-path", type=str, default="./model.pth")
    parser.add_argument("--train-data-dir", type=str, default="/home/karan/fmri")
    parser.add_argument("--eval-data-dir", type=str, default="/home/karan/fmri")
    parser.add_argument("--test-data-dir", type=str, default="/home/karan/fmri")
    parser.add_argument("--eval-only", action='store_true', default=False)
    parser.add_argument("--test-only", action='store_true', default=False)

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--optim-steps", type=int, default=1500)
    args = parser.parse_args()
    return args

def context_loss(input_images, generated_images, masks, weighted=True):
    return T.sum((input_images-generated_images)**2 * masks)

def inpaint(args):
    print("Starting inpainting ...")
    dataset = datasets.RandomPatchDataset(args.train_data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    input_image = load(args.input_image)

    # Loading trained GAN model
    saved_gan = T.load(args.gan_path)
    generator = models.Generator(args)
    discriminator = models.Discriminator(args)
    generator.load_state_dict(saved_gan["state_dict_G"])
    discriminator.load_state_dict(saved_gan["state_dict_D"])

    for i, (input_images, masks) in tqdm.tqdm(enumerate(dataloader)):
        input_images, masks = input_images.cuda(), masks.cuda()
        z_optimum = Tensor(np.random.normal(0, 1, (args.batch_size,args.latent_dim,)))
        optimizer_inpaint = optim.Adam(z_optimum)

        for epoch in tqdm.tqdm(args.optim_steps):
            optimizer_inpaint.zero_grad()
            generated_images = generator(z_optimum)
            discriminator_opinion = discriminator(generated_images)
            inpaint_loss = context_loss(input_images, generated_images, masks) + \
                    args.prior_weight*T.log(1-discriminator_opinion)
            inpaint_loss.backward()
            optimizer_inpaint.step()
    
        # save outputs
        for img_id in range(args.batch_size):
            save_image(input_images[img_id], "images/input_{}_{}.png".format(i,img_id), normalize=True)
            save_image(generated_images[img_id], "images/output_{}_{}.png".format(i,img_id), normalize=True)

    return (generator, discriminator)

if __name__ == "__main__":
    args = get_args()
    inpaint(args)
