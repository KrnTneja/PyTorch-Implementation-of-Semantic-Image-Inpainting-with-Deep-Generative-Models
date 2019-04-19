import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model-path", type=str, default="./checkpoints/model.pth")
    parser.add_argument("--new-model-path", type=str, default="./checkpoints/model.pth")
    parser.add_argument("--train-data-dir", type=str, default="../data/")
    parser.add_argument("--eval-data-dir", type=str, default="../data/")
    parser.add_argument("--test-data-dir", type=str, default="../data/")
    parser.add_argument("--eval-only", action='store_true', default=False)
    parser.add_argument("--test-only", action='store_true', default=False)

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--d-steps", type=int, default=1)
    parser.add_argument("--sample-interval", type=int, default=1)
    args = parser.parse_args()
    return args

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_checkpoint(state, filename):
    torch.save(state, filename)

def train(args):
    print("Starting training ...")
    epoch = 0
    dataset = datasets.MRIImages(args.train_data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # Tensor = torch.cuda.FloatTensor

    generator = models.Generator(args).cuda()
    discriminator = models.Discriminator(args).cuda()

    adversarial_loss = torch.nn.BCELoss()
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
            # Adversarial ground truths
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
            for _ in range(args.d_steps):
                optimizer_D.zero_grad()
                discriminator_opinion_real = discriminator(real_images)
                discriminator_opinion_fake = discriminator(gen_imgs.detach())
                real_loss = adversarial_loss(discriminator_opinion_real, valid)
                fake_loss = adversarial_loss(discriminator_opinion_fake, fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
            total_D_loss += d_loss.cpu().detach().numpy()

            if epoch % args.sample_interval == 0 and i % (len(dataloader)/5) == 0:
                save_image(gen_imgs.data[0,0], "images/{}_{}.png".format(str(epoch).zfill(len(str(args.epochs))), 
                                                                         str(i).zfill(len(str(len(dataloader))))), normalize=True)

        print(
            "[Epoch {}/{}] \t[D loss: {:.3f}] \t[G loss: {:.3f}]".format(
                epoch, args.epochs, total_D_loss, total_G_loss)
        )
    
        save_checkpoint({"epoch": epoch,
                         "state_dict_G": generator.state_dict(),
                         "state_dict_D": discriminator.state_dict(),
                         "optimizer_G": optimizer_G.state_dict(),
                         "optimizer_D": optimizer_D.state_dict()
                        }, args.new_model_path)
        # save_checkpoint({"epoch": epoch,
                         # "state_dict_G": generator.state_dict(),
                         # "state_dict_D": discriminator.state_dict(),
                         # "optimizer_G": optimizer_G.state_dict(),
                         # "optimizer_D": optimizer_D.state_dict()
                        # }, "checkpoints/{epoch}.pth".format(epoch=epoch))

    return (generator, discriminator)

def save_output(img_arr, filename, scan_name, scan_slice):
    img_arr = img_arr[0].transpose(1,2).transpose(2,3).transpose(0,1).transpose(1,2)
    img_arr = img_arr.cpu().detach().numpy()
    io.savemat(filename, {"mri_image": img_arr, "scan_name":scan_name, "slice":scan_slice})
    
def evaluate(args):
    with torch.no_grad():
        dataset = datasets.MRIImagesTest(args.test_data_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        model = get_model(args).eval()
        loss_function = Loss1()
    
        if os.path.isfile(args.model_path): 
            saved_state = torch.load(args.model_path)
            model.load_state_dict(saved_state["state_dict"])
    
        total_loss = 0.0
        batch = 0
        for inputs, targets in tqdm.tqdm(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss
            for i in range(p1images.shape[0]):
                save_output(output[i], "outputs/output_{}.mat".format(batch*args.batch_size+i), 
                            scan_name, scan_slice) #.cpu().detach().numpy()
            batch = batch+1
        print("Average Loss: {:.3f}".format(total_loss/len(dataset)))

def test(args):
    pass

if __name__ == "__main__":
    args = get_arguments()
    if not (args.eval_only or args.test_only):
        trained_models = train(args)
    if args.eval_only:
        evaluate(args)
    if args.test_only:
        test(args)

