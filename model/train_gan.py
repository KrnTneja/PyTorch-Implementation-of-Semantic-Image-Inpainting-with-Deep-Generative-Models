import torch
from torch.utils.data import Dataset, DataLoader
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

class Loss1(torch.nn.Module):
    def __init__(self):
        super(Loss1, self).__init__()

    def forward(self, output, target):
        return ??

class Loss2(torch.nn.Module):
    def __init__(self):
        super(Loss2, self).__init__()

    def forward(self, output, target):
        return ??

class SNR(torch.nn.Module):
    def __init__(self):
        super(SNR, self).__init__()

    def forward(self, output, target):
        return torch.mean((target-np.mean(target))**2)/torch.mean(((output - target)**2))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-path", type=str, default="./model.pth")
    parser.add_argument("--new-model-path", type=str, default="./model.pth")
    parser.add_argument("--train-data-dir", type=str, default="/home/karan/fmri")
    parser.add_argument("--eval-data-dir", type=str, default="/home/karan/fmri")
    parser.add_argument("--test-data-dir", type=str, default="/home/karan/fmri")
    parser.add_argument("--eval-only", action='store_true', default=False)
    parser.add_argument("--test-only", action='store_true', default=False)

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    args = parser.parse_args()
    return args

def get_model(args):
    model = models.MRINet().cuda().train()
    return model

def save_checkpoint(state, filename):
    torch.save(state, filename)

def train(args):
    print("Starting training ...")
    epoch = 0
    dataset = datasets.MRIImages(args.train_data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    generator = models.Generator(args)
    discriminator = models.Discriminator(args)

    adversarial_loss = torch.nn.BCELoss().cuda()
    optimizer_G = optim.Adam(generator.parameters())
    optimizer_D = optim.Adam(discriminator.parameters())

    if os.path.isfile(args.model_path): 
        saved_state = torch.load(args.model_path)
        epoch = saved_state['epoch']
        generator.load_state_dict(saved_state["state_dict_G"])
        discriminator.load_state_dict(saved_state["state_dict_D"])
        optimizer.load_state_dict(saved_state["optimizer"])

    while epoch < args.epochs:
        epoch = epoch+1

        for i, (imgs, _) in tqdm.tqdm(enumerate(dataloader)):
            inputs, targets = inputs.cuda(), targets.cuda()

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
            real_imgs = Variable(imgs.type(Tensor))

            #  Train Generator
            optimizer_G.zero_grad()
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            #  Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
    
        batches_done = epoch * len(dataloader) + i

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        save_checkpoint({"epoch": epoch,
                         "state_dict_G": generator.state_dict(),
                         "state_dict_D": discriminator.state_dict(),
                         "optimizer": optimizer.state_dict()
                        }, args.new_model_path)
        save_checkpoint({"epoch": epoch,
                         "state_dict_G": generator.state_dict(),
                         "state_dict_D": discriminator.state_dict(),
                         "optimizer": optimizer.state_dict()
                        }, "checkpoints/{epoch}.pth".format(epoch=epoch))

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

