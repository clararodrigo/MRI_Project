import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import wandb
import pprint
import matplotlib
import matplotlib.pyplot as plt
import torchvision.utils as vutils

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

def init_dataset(bs):
    workers = 2
    image_size = (64,64)
    dataroot = r'C:\Users\Xiaowei\Desktop\Clara\CycleGAN\Datasets\mri2sos_dataset'
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

def training_sweep():
  with wandb.init(config=sweep_config):
    config = wandb.config
    # init dataloaders with the batch size
    dataloader_mri, dataloader_sos = init_dataset(1)

    # init generators and discriminators
    G_A2B = ResnetGenerator(input_nc=p.n_channels,output_nc=p.n_channels,ngf=p.ngf,norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=config.gen_n_blocks, n_downsampling=config.gen_n_downs, padding_type='reflect').to(device)
    G_B2A = ResnetGenerator(input_nc=p.n_channels,output_nc=p.n_channels,ngf=p.ngf,norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=config.gen_n_blocks, n_downsampling=config.gen_n_downs, padding_type='reflect').to(device)
    D_A = NLayerDiscriminator(input_nc=p.n_channels,ndf=p.ndf,n_layers=config.dis_n_downs, norm_layer=nn.BatchNorm2d).to(device)
    D_B = NLayerDiscriminator(input_nc=p.n_channels,ndf=p.ndf,n_layers=config.dis_n_downs, norm_layer=nn.BatchNorm2d).to(device)

    # init optimizers
    optimizer_G_A2B = torch.optim.Adam(G_A2B.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_G_B2A = torch.optim.Adam(G_B2A.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))

    # Training Loop
    disc_A_losses, disc_B_losses, fool_disc_A_losses, fool_disc_B_losses,  cycle_A_losses,  cycle_B_losses,  id_A_losses, id_B_losses = train_epoch(config.epochs, G_A2B=G_A2B, G_B2A=G_B2A, D_A=D_A, D_B=D_B, optimizer_G_A2B=optimizer_G_A2B, optimizer_G_B2A=optimizer_G_B2A, optimizer_D_A=optimizer_D_A, optimizer_D_B=optimizer_D_B, dataloader_mri=dataloader_mri, dataloader_sos=dataloader_sos)
    
    disc_A_losses = torch.stack(disc_A_losses).tolist()
    disc_B_losses = torch.stack(disc_B_losses).tolist()
    fool_disc_A_losses = torch.stack(fool_disc_A_losses).tolist()
    fool_disc_B_losses = torch.stack(fool_disc_B_losses).tolist()
    cycle_A_losses = torch.stack(cycle_A_losses).tolist()
    cycle_B_losses = torch.stack(cycle_B_losses).tolist()
    id_A_losses = torch.stack(id_A_losses).tolist()
    id_B_losses = torch.stack(id_B_losses).tolist()

    G_A2B.cpu()
    G_B2A.cpu()
    D_A.cpu()
    D_B.cpu()
    
    
    # Calculating averages
    avg_disc_A = np.average(disc_A_losses[-100:]) 
    avg_disc_B = np.average(disc_B_losses[-100:])
    avg_fool_disc_A = np.average(fool_disc_A_losses[-100:])
    avg_fool_disc_B = np.average(fool_disc_B_losses[-100:])
    avg_cycle_A = np.average(cycle_A_losses[-100:])
    avg_cycle_B = np.average(cycle_B_losses[-100:])
    avg_id_A = np.average(id_A_losses[-100:])
    avg_id_B = np.average(id_B_losses[-100:])

    g_losses = avg_fool_disc_A+avg_fool_disc_B+avg_cycle_A+avg_cycle_B+avg_id_A+avg_id_B
    d_losses = avg_disc_A+avg_disc_B
    print("___________________________________________________________________")
    print("Generator losses: " + str(g_losses))
    print("Discriminator losses: " + str(d_losses))
    print("___________________________________________________________________")

    wandb.log({
            "total_g" : g_losses,
            "total_d" : d_losses
        })
    
def train_epoch(epochs, G_A2B, G_B2A, D_A, D_B, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B, dataloader_mri, dataloader_sos, old=True):

  print("Starting New Epoch...")

  disc_A_losses = []
  disc_B_losses = []
  fool_disc_A_losses = []
  fool_disc_B_losses = []
  cycle_A_losses = []
  cycle_B_losses = []
  id_A_losses = []
  id_B_losses = []

  for epoch in range(epochs):
    iters=0

    # For each batch in the dataloader
    for i,(data_mri, data_sos) in enumerate(zip(dataloader_mri, dataloader_sos),0):
    
      # Set model input
      a_real = data_mri[0][:,0,:,:].unsqueeze(1).to(device)
      b_real = data_sos[0][:,0,:,:].unsqueeze(1).to(device)

      # Generated images
      b_fake = G_A2B(a_real)
      a_rec = G_B2A(b_fake)
      a_fake = G_B2A(b_real)
      b_rec = G_A2B(a_fake)

      # CALCULATE DISCRIMINATORS LOSSES
      # Discriminator A
      optimizer_D_A.zero_grad()
      if((iters > 0 ) and old and iters % 3 == 0):
        rand_int = random.randint(1, old_a_fake.shape[0]-1)
        Disc_loss_A = LSGAN_D(D_A(a_real), D_A(old_a_fake[rand_int-1:rand_int].detach()))
        disc_A_losses.append(Disc_loss_A)
      else:
        Disc_loss_A = LSGAN_D(D_A(a_real), D_A(a_fake.detach()))
        disc_A_losses.append(Disc_loss_A)

      Disc_loss_A.backward()
      optimizer_D_A.step()
      
      # Discriminator B
      optimizer_D_B.zero_grad()
      if((iters > 0) and old and iters % 3 == 0):
        rand_int = random.randint(1, old_b_fake.shape[0]-1)
        Disc_loss_B = LSGAN_D(D_B(b_real), D_B(old_b_fake[rand_int-1:rand_int].detach()))
        disc_B_losses.append(Disc_loss_B)
      else:
        Disc_loss_B = LSGAN_D(D_B(b_real), D_B(b_fake.detach()))
        disc_B_losses.append(Disc_loss_B)

      Disc_loss_B.backward()
      optimizer_D_B.step() 

      # Generator
      optimizer_G_A2B.zero_grad()
      optimizer_G_B2A.zero_grad()
      

      # CALCULATE GENERATORS LOSSES
      Fool_disc_loss_A2B = LSGAN_G(D_B(b_fake))
      fool_disc_A_losses.append(Fool_disc_loss_A2B)
      Fool_disc_loss_B2A = LSGAN_G(D_A(a_fake))
      fool_disc_B_losses.append(Fool_disc_loss_B2A)

      # Cycle Consistency
      Cycle_loss_A = criterion_Im(a_rec, a_real)*5
      cycle_A_losses.append(Cycle_loss_A)
      Cycle_loss_B = criterion_Im(b_rec, b_real)*5
      cycle_B_losses.append(Cycle_loss_B)

      # Identity loss
      Id_loss_B2A = criterion_Im(G_B2A(a_real), a_real)*10
      Id_loss_A2B = criterion_Im(G_A2B(b_real), b_real)*10
      id_A_losses.append(Id_loss_A2B)
      id_B_losses.append(Id_loss_B2A)

      # generator losses
      Loss_G = Fool_disc_loss_A2B+Fool_disc_loss_B2A+Cycle_loss_A+Cycle_loss_B+Id_loss_B2A+Id_loss_A2B

      # Backward propagation
      Loss_G.backward()
      
      # Optimisation step
      optimizer_G_A2B.step()
      optimizer_G_B2A.step()
      
      if(iters == 0):
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
      with torch.no_grad():
        del a_real, b_real, a_fake, b_fake, a_rec, b_rec
        torch.cuda.empty_cache()

      print('Iter finished: '+str(iters))
    with torch.no_grad():
        del old_a_fake, old_b_fake
        torch.cuda.empty_cache()
                
  return disc_A_losses, disc_B_losses, fool_disc_A_losses, fool_disc_B_losses,  cycle_A_losses,  cycle_B_losses,  id_A_losses, id_B_losses


if __name__ ==  '__main__':
  print("_________________________________________________")
  if torch.cuda.is_available():
      print("The code will run on GPU.")
      torch.cuda.manual_seed_all(999)
  else:
      print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  p = Parameters(bs=1, n_channels=1, ngf=64, ndf=64, size=64, gen_n_down=2, gen_n_blocks=6, dis_n_down=2, lr=0.0002, beta1=0.5)
  criterion_Im = torch.nn.L1Loss()

  wandb.login()

  sweep_config = {
    "method" : "random"}

  metric = {
      'name': 'total_g',
      'goal': 'minimize'   
      }
  sweep_config['metric'] = metric

  parameters_dict = {
      'epochs': {
          'value': 2},
      'learning_rate': {
          # a flat distribution between 0 and 0.1
          'distribution': 'uniform',
          'min': 0,
          'max': 0.1
        },
      'beta1': {
          'min' : 0.7,
          'max' : 1.0
        },
      'gen_n_downs':{
        'values' : [1,2,3,4]},
      'gen_n_blocks' : {
        'values' : [1,2,3,4,5,6]},
      'dis_n_downs' : {
        'values' : [1,2,3,4]}
      }
  sweep_config['parameters'] = parameters_dict
  #sweep_id = wandb.sweep(sweep_config, project='mri2sos_sweeps')
  # visualize parameters
  pprint.pprint(sweep_config)

  print("Starting agent")
  wandb.agent("ydeinf48", project="mri2sos_sweeps", function=training_sweep, count=30)
  print("Finished agent")