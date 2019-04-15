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
    model = get_model(args)
    loss_function = Loss1()
    final_loss_function = Loss2()
    optimizer = optim.Adam(model.parameters())

    if os.path.isfile(args.model_path): 
        saved_state = torch.load(args.model_path)
        epoch = saved_state['epoch']
        model.load_state_dict(saved_state["state_dict"])
        optimizer.load_state_dict(saved_state["optimizer"])

    while epoch < args.epochs:
        epoch = epoch+1
        total_loss = 0.0
        total_final_loss = 0.0
        for inputs, targets in tqdm.tqdm(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            final_loss = final_loss_function(outputs, targets)
            total_loss += loss
            total_final_loss += final_loss
            loss.backward()
            optimizer.step()

        print("Epoch: {} Loss: {:.3f} Final loss: {:.3f}".format(
            epoch, total_loss/len(dataset), total_final_loss/len(dataset)))
        save_checkpoint({"epoch": epoch,
                         "state_dict": model.state_dict(),
                         "optimizer": optimizer.state_dict()
                        }, args.new_model_path)
    return model

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
        trained_model = train(args)
    if args.eval_only:
        evaluate(args)
    if args.test_only:
        test(args)

