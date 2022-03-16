import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pylab as plt
import wandb
from goto import with_goto

import glob
from glob import glob
import nibabel as nb

from dataclasses import dataclass
from ResnetGenerator import ResnetGenerator
from NLayerDiscriminator import NLayerDiscriminator

@dataclass
class Parameters:
    bs : int
    n_channels : int
    ngf : int
    ndf : int
    size : int
    gen_n_down : int
    gen_n_blocks : int
    dis_n_down : int
    lr : float
    beta1 : float

def init_dataset(bs, test = False):
  workers = 2
  image_size = (64,64)
  dataroot_MRI_SoS = r'C:\Users\Xiaowei\Desktop\Clara\CycleGAN\Datasets\same_patch_dataset'
  dataroot_CT = r"C:\Users\Xiaowei\Desktop\Clara\CycleGAN\Datasets\CT_dataset\A\ct_img1.png"

  datasets_train = dset.ImageFolder(root=dataroot,
                          transform=transforms.Compose([
                              transforms.Resize(image_size),
                              transforms.CenterCrop(image_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0, 0, 0), (1, 1, 1)),
                              ]))

  from torch.utils import data
  idx = [i for i in range(len(datasets_train)) if datasets_train.imgs[i][1] != datasets_train.class_to_idx['B']]
  mri_subset = data.Subset(datasets_train, idx)
  dataloader_mri = torch.utils.data.DataLoader(mri_subset, batch_size=bs,
                                          shuffle=True, num_workers=workers)
  idx = [i for i in range(len(datasets_train)) if datasets_train.imgs[i][1] != datasets_train.class_to_idx['A']]
  sos_subset = data.Subset(datasets_train, idx)

  dataloader_sos = torch.utils.data.DataLoader(sos_subset, batch_size=bs,
                                          shuffle=True, num_workers=workers)
  
  return dataloader_mri, dataloader_sos

def init_models(p):
    G_A2B = ResnetGenerator(input_nc=p.n_channels,output_nc=p.n_channels,ngf=p.ngf,norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=p.gen_n_blocks, n_downsampling=p.gen_n_down, padding_type='reflect')
    G_B2A = ResnetGenerator(input_nc=p.n_channels,output_nc=p.n_channels,ngf=p.ngf,norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=p.gen_n_blocks, n_downsampling=p.gen_n_down, padding_type='reflect')
    D_A = NLayerDiscriminator(input_nc=p.n_channels,ndf=p.ndf,n_layers=p.dis_n_down, norm_layer=nn.BatchNorm2d)
    D_B = NLayerDiscriminator(input_nc=p.n_channels,ndf=p.ndf,n_layers=p.dis_n_down, norm_layer=nn.BatchNorm2d)

    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=p.lr, betas=(p.beta1, 0.999))
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=p.lr, betas=(p.beta1, 0.999))

    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=p.lr, betas=(p.beta1, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=p.lr, betas=(p.beta1, 0.999))

    return G_A2B, G_B2A, D_A, D_B, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B

def LSGAN_D(real, fake):
  return (torch.mean((real - 1)**2) + torch.mean(fake**2))
def LSGAN_G(fake):
  return  torch.mean((fake - 1)**2)

def plot_images_test(dataloader_mri, dataloader_sos): 
    batch_a_test = next(iter(dataloader_mri))[0][:,0,:,:].unsqueeze(1).to(device)
    real_a_test = batch_a_test.cpu().detach()
    fake_b_test = G_A2B(batch_a_test).cpu().detach()

    plt.subplots(1,4, figsize=(10,80))
    plt.subplot(1,4,1)
    plt.imshow(np.transpose(vutils.make_grid((real_a_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Real MRIs")
          
    plt.subplot(1,4,2)
    plt.imshow(np.transpose(vutils.make_grid((fake_b_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Fake SoS images")

    batch_b_test = next(iter(dataloader_sos))[0][:,0,:,:].unsqueeze(1).to(device)
    real_b_test = batch_b_test.cpu().detach()
    fake_a_test = G_B2A(batch_b_test ).cpu().detach()
    
    plt.subplot(1,4,3)
    plt.imshow(np.transpose(vutils.make_grid((real_b_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Real SoS images")

    plt.subplot(1,4,4)
    plt.imshow(np.transpose(vutils.make_grid((fake_a_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Fake MRIs")
    matplotlib.pyplot.show()

def save_models(G_A2B, G_B2A, D_A, D_B, name):
  torch.save(G_A2B, name+"_G_A2B.pt")
  torch.save(G_B2A,  name+"_G_B2A.pt")
  torch.save(D_A,  name+"_D_A.pt")
  torch.save(D_B, name+"_D_B.pt")
def load_models( name):
  G_A2B=torch.load(name+'_G_A2B.pt', map_location=torch.device('cpu'))
  G_B2A=torch.load(name+"_G_B2A.pt",map_location=torch.device('cpu'))
  D_A=torch.load(name+"_D_A.pt", map_location=torch.device('cpu'))
  D_B=torch.load(name+"_D_B.pt", map_location=torch.device('cpu'))
  return G_A2B, G_B2A, D_A, D_B 

