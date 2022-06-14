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

path = '/home/oab18/Projects/MRI_Project/Dataset/noisy_4/train/'
dataset_mri = torch.load(path+'mri_dataset.pt')
dataset_sos = torch.load(path+'sos_dataset.pt')
print(len(dataset_mri))
for count in range(1,500,100):
    test_mris.append(dataset_mri[count])
    test_soss.append(dataset_sos[count])
# [73:193,66:236]
print("Number of test images: "+str(len(test_mris)))

test_fake_mris = []
test_fake_soss = []
test_rec_mris = []
test_rec_soss = []

G_A2B.to(device)
G_B2A.to(device)
D_A.to(device)
D_B.to(device)
for i in range(len(test_mris)):
    real_mri = torch.Tensor(test_mris[i]).unsqueeze(0).unsqueeze(0)
    fake_sos = G_A2B(real_mri.to(device)).cpu().detach().numpy()
    rec_mri = G_B2A(torch.Tensor(fake_sos).to(device)).cpu().detach().numpy()

    real_sos = torch.Tensor(test_soss[i]).unsqueeze(0).unsqueeze(0)
    fake_mri = G_B2A(real_sos.to(device)).cpu().detach().numpy()
    rec_sos = G_A2B(torch.Tensor(fake_mri).to(device)).cpu().detach().numpy()

    test_fake_mris.append(fake_mri)
    test_fake_soss.append(fake_sos)
    test_rec_mris.append(rec_mri)
    test_rec_soss.append(rec_sos)

    with torch.no_grad():
        real_mri.detach().cpu()
        real_sos.detach().cpu()
n=2
print(test_soss[n].shape, test_fake_soss[n].shape)
# diff_mri = (test_mris.cpu().detach().numpy()-test_fake_mris[0,0,:,:])[73:193,66:216]
# diff_sos = (test_soss.cpu().detach().numpy()-test_fake_soss[0,0,:,:])[73:193,66:216]
# plt.imshow(diff[n]);plt.colorbar();plt.show()


test_mris_np = []
test_soss_np = []
test_fake_mris_np = []
test_fake_soss_np = []
for i in range(len(test_mris)):
    test_mris_np.append(test_mris[i].cpu().detach().numpy())
    test_soss_np.append(test_soss[i].cpu().detach().numpy())

avg_realmri = np.mean(np.array(test_mris_np[:]),axis=0).flatten()
avg_fakemri = np.mean(np.array(test_fake_mris[:]),axis=0).flatten()
avg_realsos = np.mean(np.array(test_soss_np[:]),axis=0).flatten()
avg_fakesos = np.mean(np.array(test_fake_soss[:]),axis=0).flatten()
print(np.corrcoef([avg_realmri, avg_fakemri]))
print(np.corrcoef([avg_realsos, avg_fakesos]))

real_mris = torch.mean(torch.Tensor(test_mris)).flatten()
rec_mris = torch.mean(torch.Tensor(test_rec_mris)).flatten()

# plt.imshow(test_mris[1]);plt.colorbar();plt.show()
# xy = np.vstack([real_mris, rec_mris])
# z = gaussian_kde(xy)(xy)
# plt.scatter(real_mris,rec_mris,c=z);plt.show()

