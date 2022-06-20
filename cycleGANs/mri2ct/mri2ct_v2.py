import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import wandb

import glob
from glob import glob
import nibabel as nb
import imageio

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

def init_dataset(bs, test=False):
    if(test == False): path = '/home/oab18/Projects/MRI_Project/Dataset/ct_skulls/train/'
    if(test == True): path = '/home/oab18/Projects/MRI_Project/Dataset/ct_skulls/test/'
    dataset_mri = torch.load(path+'mri_dataset.pt')
    dataset_ct = torch.load(path+'ct_dataset.pt')

    workers = 2
    image_size = (64,64)
    # dataroot = r'/home/oab18/Projects/MRI_Project/Dataset/mri2ct2sos_dataset/'

    from torch.utils import data
    dataloader_mri = torch.utils.data.DataLoader(dataset_mri, batch_size=bs,
                                            shuffle=True, num_workers=2)

    dataloader_ct = torch.utils.data.DataLoader(dataset_ct, batch_size=bs,
                                            shuffle=True, num_workers=2)
    
    return dataloader_mri, dataloader_ct

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

def plot_images_test(dataloader_mri, dataloader_ct, epoch): 
    batch_a_test = next(iter(dataloader_mri))[0].unsqueeze(0).unsqueeze(0).to(device)
    real_a_test = batch_a_test.cpu().detach()
    fake_b_test = G_A2B(batch_a_test).cpu().detach()

    batch_b_test = next(iter(dataloader_ct))[0].to(device)
    real_b_test = batch_b_test.cpu().detach()
    fake_a_test = G_B2A(batch_b_test ).cpu().detach()

    fake_a_error = (real_a_test - fake_a_test)[0,0,:,:]
    fake_b_error = (real_b_test - fake_b_test)[0,0,:,:]

    plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(fake_a_error,cmap='PiYG'); plt.colorbar()
    plt.clim(-1,1)
    plt.title('Fake MRI MAE')
    plt.subplot(1,2,2)
    plt.imshow(fake_b_error,cmap='PiYG'); plt.colorbar()
    plt.clim(-1,1)
    plt.title('Fake CT MAE')
    # plt.show(block=False)
    plt.savefig('/home/oab18/Projects/MRI_Project/cycleGANs/Images/training_'+name+'/err_epoch'+str(epoch)+'.png')
    

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

def train(epochs, G_A2B, G_B2A, D_A, D_B, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B, dataloader_mri, dataloader_sos, old=True):

  print(wandb.config)  

  # Training Loop

  # Lists to keep track of progress
  img_list = []
  G_losses = []
  D_A_losses = []
  D_B_losses = []


  iters=0
  FDL_A2B = []
  FDL_B2A = []
  CL_A = []
  CL_B = []
  ID_B2A = []
  ID_A2B = []
  disc_A = []
  disc_B = []


  FDL_A2B_t = []
  FDL_B2A_t = []
  CL_A_t = []
  CL_B_t = []
  ID_B2A_t = []
  ID_A2B_t = []
  disc_A_t = []
  disc_B_t = []

  print("Starting Training Loop...")
  # For each epoch
  for epoch in range(epochs):

    # For each batch in the d0ataloader
    for i,(data_mri, data_ct) in enumerate(zip(dataloader_mri, dataloader_ct),0):
        # Set model input
        a_real = data_mri[0].unsqueeze(0).unsqueeze(0).to(device)
        b_real = data_ct[0].to(device)
      
        # Generated images
        b_fake = G_A2B(a_real)
        a_rec = G_B2A(b_fake)
        a_fake = G_B2A(b_real)
        b_rec = G_A2B(a_fake)

        # CALCULATE DISCRIMINATORS LOSSES
        # Discriminator A
        optimizer_D_A.zero_grad()
        if((iters > 0 or epoch > 0) and old and iters % 3 == 0):
          if(np.random.rand() > 0.95): 
            Disc_loss_A = torch.zeros(1,1)
          else: 
            rand_int = random.randint(1, old_a_fake.shape[0]-1)
            Disc_loss_A = LSGAN_D(D_A(a_real), D_A(old_a_fake[rand_int-1:rand_int].detach()))
          D_A_losses.append(Disc_loss_A.item())

        else:
          if(np.random.rand() > 0.95): 
            Disc_loss_A = torch.Tensor(0)
          else: 
            Disc_loss_A = LSGAN_D(D_A(a_real), D_A(a_fake.detach()))
          D_A_losses.append(Disc_loss_A.item())

        
        Disc_loss_A.backward()
        optimizer_D_A.step()

        
        # Discriminator B
        optimizer_D_B.zero_grad()
        if((iters > 0 or epoch > 0) and old and iters % 3 == 0):
          if(np.random.rand() > 0.95): 
            Disc_loss_B = torch.zeros(1,1)
          else: 
            rand_int = random.randint(1, old_b_fake.shape[0]-1)
            Disc_loss_B =  LSGAN_D(D_B(b_real), D_B(old_b_fake[rand_int-1:rand_int].detach()))
          D_B_losses.append(Disc_loss_B.item())
        else:
          if(np.random.rand() > 0.95):
            Disc_loss_B = torch.zeros(1,1)
          else:
            Disc_loss_B =  LSGAN_D(D_B(b_real), D_B(b_fake.detach()))
          D_B_losses.append(Disc_loss_B.item())

        Disc_loss_B.backward()
        optimizer_D_B.step()   

        # Generator
        optimizer_G_A2B.zero_grad()
        optimizer_G_B2A.zero_grad()


        # CALCULATE GENERATORS LOSSES
        Fool_disc_loss_A2B = LSGAN_G(D_B(b_fake))
        Fool_disc_loss_B2A = LSGAN_G(D_A(a_fake))

        # Cycle Consistency    both use the two generators
        Cycle_loss_A = criterion_Im(a_rec, a_real)*5
        Cycle_loss_B = criterion_Im(b_rec, b_real)*5

        # Identity loss
        Id_loss_B2A = criterion_Im(G_B2A(a_real), a_real)*10
        Id_loss_A2B = criterion_Im(G_A2B(b_real), b_real)*10

        # generator losses
        Loss_G = Fool_disc_loss_A2B+Fool_disc_loss_B2A+Cycle_loss_A+Cycle_loss_B+Id_loss_B2A+Id_loss_A2B
        G_losses.append(Loss_G)

        wandb.log({
            "disc_loss_A2B": Fool_disc_loss_A2B,
            "disc_loss_B2A": Fool_disc_loss_B2A,
            "Cycle_loss_A": Cycle_loss_A,
            "Cycle_loss_B": Cycle_loss_B,
            "id_loss_A2B": Id_loss_A2B,
            "id_loss_B2A": Id_loss_B2A,
            "total_g": Loss_G,
            "total_d": Fool_disc_loss_A2B + Fool_disc_loss_B2A
        })

        # Backward propagation
        Loss_G.backward()
        
        # Optimisation step
        optimizer_G_A2B.step()
        optimizer_G_B2A.step()

        FDL_A2B.append(Fool_disc_loss_A2B)
        FDL_B2A.append(Fool_disc_loss_B2A)
        CL_A.append(Cycle_loss_A)
        CL_B.append(Cycle_loss_B)
        ID_B2A.append(Id_loss_B2A)
        ID_A2B.append(Id_loss_A2B)
        disc_A.append(Disc_loss_A)
        disc_B.append(Disc_loss_B)

        if(iters == 0 and epoch == 0):
          old_b_fake = b_fake.clone()
          old_a_fake = a_fake.clone()
        elif (old_b_fake.shape[0] == p.bs*5 and b_fake.shape[0]==p.bs):
          rand_int = random.randint(5, 24)
          old_b_fake[rand_int-5:rand_int] = b_fake.clone()
          old_a_fake[rand_int-5:rand_int] = a_fake.clone()
        elif(old_b_fake.shape[0]< 25):
          old_b_fake = torch.cat((b_fake.clone(),old_b_fake))
          old_a_fake = torch.cat((a_fake.clone(),old_a_fake))

        iters += 1
        del a_real, b_real, a_fake, b_fake


        if iters % 50 == 0:
      
          print('[%d/%d]\tFDL_A2B: %.4f\tFDL_B2A: %.4f\tCL_A: %.4f\tCL_B: %.4f\tID_B2A: %.4f\tID_A2B: %.4f\tLoss_D_A: %.4f\tLoss_D_A: %.4f'
                      % (epoch+1, epochs, Fool_disc_loss_A2B, Fool_disc_loss_B2A,Cycle_loss_A,Cycle_loss_B,Id_loss_B2A,
                          Id_loss_A2B, Disc_loss_A.item(), Disc_loss_B.item()))
        
        

    FDL_A2B_t.append(sum(FDL_A2B)/len(FDL_A2B))
    FDL_B2A_t.append(sum(FDL_B2A)/len(FDL_B2A))
    CL_A_t.append(sum(CL_A)/len(CL_A))
    CL_B_t.append(sum(CL_B)/len(CL_B))
    ID_B2A_t.append(sum(ID_B2A)/len(ID_B2A))
    ID_A2B_t.append(sum(ID_A2B)/len(ID_A2B))
    disc_A_t.append(sum(disc_A)/len(disc_A))
    disc_B_t.append(sum(disc_B)/len(disc_B))

    FDL_A2B = []
    FDL_B2A = []
    CL_A = []
    CL_B = []
    ID_B2A = []
    ID_A2B = []
    disc_B = []
    disc_A = []

    iters = 0             
    save_models(G_A2B, G_B2A, D_A, D_B, name+'_epoch_'+str(epoch))
    if (epoch % 5 == 0):
      plot_images_test(dataloader_mri, dataloader_ct, epoch)
  return(FDL_A2B_t,FDL_B2A_t,CL_A_t,CL_B_t,ID_B2A_t,ID_A2B_t,disc_A_t,disc_B_t)


if __name__ ==  '__main__':
    if torch.cuda.is_available():
      print("The code will run on GPU.")
      torch.cuda.manual_seed_all(999)
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    p = Parameters(bs=1, n_channels=1, ngf=128, ndf=16, size=64, gen_n_down=2, gen_n_blocks=6, dis_n_down=2, lr=0.0002, beta1=0.5)
    criterion_Im = torch.nn.L1Loss()

    dataloader_mri, dataloader_ct = init_dataset(p.bs, False)
    test_dataloader_mri, test_dataloader_ct = init_dataset(p.bs, True)
    
    wandb.login()
    name = 'ct_skulls_2'
    wandb.init(project='mri2ct2sos', entity='clara_rg_')

    epochs=100
    
    G_A2B, G_B2A, D_A, D_B, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B = init_models(p)
    # G_A2B, G_B2A, D_A, D_B = load_models('ct_1_epoch_0')
    # optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=p.lr, betas=(p.beta1, 0.999))
    # optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=p.lr, betas=(p.beta1, 0.999))
    # optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=p.lr, betas=(p.beta1, 0.999))
    # optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=p.lr, betas=(p.beta1, 0.999))
    G_A2B.to(device)
    G_B2A.to(device)
    D_A.to(device)
    D_B.to(device)
    
    losses = train(epochs, G_A2B, G_B2A, D_A, D_B, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B, dataloader_mri, dataloader_ct, old=True)


