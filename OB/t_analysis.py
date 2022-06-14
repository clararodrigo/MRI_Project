import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pylab as plt
from scipy.stats import gaussian_kde

import glob
from glob import glob
import nibabel as nb

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed_all(42)

def load_models( name):
  G_A2B=torch.load(name+'_G_A2B.pt', map_location=torch.device('cpu'))
  G_B2A=torch.load(name+"_G_B2A.pt",map_location=torch.device('cpu'))
  D_A=torch.load(name+"_D_A.pt", map_location=torch.device('cpu'))
  D_B=torch.load(name+"_D_B.pt", map_location=torch.device('cpu'))
  return G_A2B, G_B2A, D_A, D_B

G_A2B, G_B2A, D_A, D_B = load_models('full_8_epoch_63')

import os
test_mris = []
test_soss = []
ns = []
count = 0

path = '/home/oab18/Projects/MRI_Project/Dataset/noisy_4/test/'
dataset_mri = torch.load(path+'mri_dataset.pt')
dataset_sos = torch.load(path+'sos_dataset.pt')
real_mri = torch.Tensor(dataset_mri[300]).unsqueeze(0).unsqueeze(0)
real_sos = torch.Tensor(dataset_sos[300]).unsqueeze(0).unsqueeze(0)


test_fake_mris = []
test_fake_soss = []
# test_rec_mris = []
# test_rec_soss = []

G_A2B, G_B2A, D_A, D_B = load_models('full_8_epoch_63')

G_A2B.to(device)
G_B2A.to(device)
D_A.to(device)
D_B.to(device)
for i in range(100):
    fake_sos = G_A2B(real_mri.to(device)).cpu().detach().numpy()
    fake_mri = G_B2A(real_sos.to(device)).cpu().detach().numpy()
    # rec_mri = G_B2A(torch.Tensor(fake_sos).to(device)).cpu().detach().numpy()

    # real_sos = torch.Tensor(test_soss[i]).unsqueeze(0).unsqueeze(0)
    # fake_mri = G_B2A(real_sos.to(device)).cpu().detach().numpy()
    # rec_sos = G_A2B(torch.Tensor(fake_mri).to(device)).cpu().detach().numpy()

    test_fake_mris.append(np.array(fake_mri))
    test_fake_soss.append(np.array(fake_sos))
    # test_rec_mris.append(rec_mri)
    # test_rec_soss.append(rec_sos)

with torch.no_grad():
    real_mri.detach().cpu()
    real_sos.detach().cpu()

avg_fake_sos = np.mean(test_fake_soss, axis=0)[0,0,:,:]
std_fake_sos = np.std(test_fake_soss, axis=0)[0,0,:,:]
avg_fake_mri = np.mean(test_fake_mris, axis=0)[0,0,:,:]
std_fake_mri = np.std(test_fake_mris, axis=0)[0,0,:,:]
print(np.mean(avg_fake_sos))
print(np.std(avg_fake_sos))
print(np.mean(np.array(real_sos)))
print(np.std(np.array(real_sos)))
diff_sos = []
diff_mri = []
for i in range(100):
    diff_sos.append(np.array(real_sos-test_fake_soss[i])[0,0,:,:])
    diff_mri.append(np.array(real_mri-test_fake_mris[i])[0,0,:,:])


s_sos = np.std(diff_sos, axis=0)
s_mri = np.std(diff_mri, axis=0)

# plt.figure();plt.imshow(avg_fake_mri); plt.colorbar()
# plt.figure();plt.imshow(std_fake_sos);plt.colorbar(); plt.show()

t_sos = ((real_sos-avg_fake_sos)[0,0,:,:]/(s_sos/np.sqrt(100)))
t_mri = ((real_mri-avg_fake_mri)[0,0,:,:]/(s_mri/np.sqrt(100)))
# plt.figure();plt.imshow(t_mri); plt.colorbar();plt.show()

plt.figure();plt.imshow(t_sos);plt.colorbar();
plt.figure();plt.imshow(t_mri);plt.colorbar();plt.show()

mask_sos = np.zeros(t_sos.shape)
mask_mri = np.zeros(t_mri.shape)
for x in range(len(t_sos[:,0])):
    for y in range(len(t_sos[0,:])):
        if t_sos[x,y] > 1.66:
            mask_sos[x,y] = 1

        if t_mri[x,y] > 1.66:
            mask_mri[x,y] = 1


plt.figure();plt.imshow(mask_sos); plt.colorbar();
plt.figure();plt.imshow(mask_mri); plt.colorbar();plt.show()

# plt.figure();plt.imshow((real_mri-avg_fake_mri)[0,0,:,:]);plt.colorbar();plt.show()
