import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

import numpy as np

import os
import argparse

import models
import datasets

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model-path", type=str, default="./checkpoints/model.pth")
    parser.add_argument("--train-data-dir", type=str, default="../data/")

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()
    return args

# Training script adapted from following repository: https://github.com/eriklindernoren/PyTorch-GAN

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def train(args):
    print("Starting training ...")
    epoch = 0
    dataset = datasets.GANImages(args.train_data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = models.Generator(args).cuda()
    discriminator = models.Discriminator(args).cuda()

    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters())
    optimizer_D = optim.Adam(discriminator.parameters())

    if os.path.isfile(args.model_path): 
        saved_state = torch.load(args.model_path)
        epoch = saved_state['epoch']
        generator.load_state_dict(saved_state["state_dict_G"])
        discriminator.load_state_dict(saved_state["state_dict_D"])
        optimizer_G.load_state_dict(saved_state["optimizer_G"])
        optimizer_D.load_state_dict(saved_state["optimizer_D"])
    else:
        generator.apply(weights_init_normal)
        discriminator.apply(weights_init_normal)

    while epoch < args.epochs:
        epoch = epoch+1
        total_G_loss = 0.0
        total_D_loss = 0.0

        for i, real_images in enumerate(dataloader):
            valid = torch.FloatTensor(real_images.shape[0], 1).fill_(1.0).cuda()
            fake = torch.FloatTensor(real_images.shape[0], 1).fill_(0.0).cuda()
            real_images = real_images.cuda()

            #  Train Generator
            optimizer_G.zero_grad()
            z = torch.FloatTensor(np.random.normal(0, 1, (real_images.shape[0], args.latent_dim))).cuda()
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            total_G_loss += g_loss.cpu().detach().numpy()

            #  Train Discriminator
            optimizer_D.zero_grad()
            discriminator_opinion_real = discriminator(real_images)
            discriminator_opinion_fake = discriminator(gen_imgs.detach())
            real_loss = adversarial_loss(discriminator_opinion_real, valid)
            fake_loss = adversarial_loss(discriminator_opinion_fake, fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if epoch % args.sample_interval == 0 and i % (len(dataloader)/5) == 0:
                save_image(gen_imgs.data[0,0], "images/{}_{}.png".format(str(epoch).zfill(len(str(args.epochs))), 
                                                                         str(i).zfill(len(str(len(dataloader))))), normalize=True)

        print(
            "[Epoch {}/{}] \t[D loss: {:.3f}] \t[G loss: {:.3f}]".format(
                epoch, args.epochs, total_D_loss, total_G_loss)
        )
    
        torch.save({"epoch": epoch,
                    "state_dict_G": generator.state_dict(),
                    "state_dict_D": discriminator.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D": optimizer_D.state_dict()
                   }, args.model_path)

        if epoch % args.checkpoint_interval == 0:
            torch.save({"epoch": epoch,
                        "state_dict_G": generator.state_dict(),
                        "state_dict_D": discriminator.state_dict(),
                        "optimizer_G": optimizer_G.state_dict(),
                        "optimizer_D": optimizer_D.state_dict()
                       }, "checkpoints/{epoch}.pth".format(epoch=epoch))

if __name__ == "__main__":
    args = get_arguments()
    train(args)

