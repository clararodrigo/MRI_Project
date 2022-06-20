import numpy as np
import random
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pylab as plt

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

G_A2B, G_B2A, D_A, D_B = load_models('ct_skulls_1_epoch_29')

import os
test_mris = []
test_soss = []
ns = []
count = 0

path = '/home/oab18/Projects/MRI_Project/Dataset/ct_skulls/test/'
dataset_mri = torch.load(path+'mri_dataset.pt')
dataset_sos = torch.load(path+'ct_dataset.pt')
print(len(dataset_mri))
for count in range(1,100):
    # n = random.choice(os.listdir(r'/home/oab18/Projects/MRI_Project/Dataset/noisy_2/test/A')).split('.')[0].split('g')[1]
    # if n not in ns:
    #     ns.append(n)
    #     test_mris.append(plt.imread(r'/home/oab18/Projects/MRI_Project/Dataset/noisy_2/test/A/mri_img'+str(n)+'.png'))
    #     test_soss.append(plt.imread(r'/home/oab18/Projects/MRI_Project/Dataset/noisy_2/test/B/sos_img'+str(n)+'.png'))
    #     count += 1
    #     print('x', end="")
    
    test_mris.append(dataset_mri[count])
    test_soss.append(dataset_sos[count])

# for i in range(len(test_soss)):
#     plt.figure(); plt.imshow(test_soss[i][0,0,:,:]); plt.show()

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

    real_sos = torch.Tensor(test_soss[i])#.unsqueeze(0).unsqueeze(0)
    fake_mri = G_B2A(real_sos.to(device)).cpu().detach().numpy()
    rec_sos = G_A2B(torch.Tensor(fake_mri).to(device)).cpu().detach().numpy()

    test_fake_mris.append(fake_mri)
    test_fake_soss.append(fake_sos)
    test_rec_mris.append(rec_mri)
    test_rec_soss.append(rec_sos)

    with torch.no_grad():
        real_mri.detach().cpu()
        real_sos.detach().cpu()

# plot real and fake MRIs
# plt.rcParams.update({'font.size': 16})
# for n in range(5):
#     plt.subplots(1,2)
#     plt.subplot(1,2,1);plt.imshow(test_mris[n]); plt.title('Real MRI'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
#     plt.subplot(1,2,2);plt.imshow(test_fake_mris[n][0,0,:,:]); plt.title('Fake MRI'); plt.colorbar(); plt.clim(0,1); plt.axis('off')

#     plt.subplots(1,2)
#     plt.subplot(1,2,1);plt.imshow(test_soss[n][0,0,:,:]); plt.title('Real SoS'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
#     plt.subplot(1,2,2);plt.imshow(test_fake_soss[n][0,0,:,:]); plt.title('Fake SoS'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
# plt.show()
print(test_mris[1].shape)


n=2
test_mris[n] = test_mris[n]/torch.max(test_mris[n])
test_fake_mris[n] = test_fake_mris[n]/np.max(test_fake_mris[n])
test_rec_mris[n] = test_rec_mris[n]/np.max(test_rec_mris[n])
plt.subplots(2,4,figsize=(8,2));
plt.subplot(2,4,1);plt.imshow(test_mris[n]); plt.title('Real MRI'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
plt.subplot(2,4,2);plt.imshow(test_fake_mris[n][0,0,:,:]); plt.title('Artificial MRI'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
plt.subplot(2,4,3);plt.imshow(test_rec_mris[n][0,0,:,:]); plt.title('Rec. MRI'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
plt.subplot(2,4,4);plt.imshow(test_mris[n]-test_rec_mris[n][0,0,:,:], cmap='PiYG');plt.colorbar();plt.title('Rec. Error'); plt.clim(-0.15,0.15); plt.axis('off')
plt.subplot(2,4,5);plt.imshow(test_soss[n][0,0,:,:]); plt.title('Real SoS'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
plt.subplot(2,4,6);plt.imshow(test_fake_soss[n][0,0,:,:]); plt.title('Artificial SoS'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
plt.subplot(2,4,7);plt.imshow(test_rec_soss[n][0,0,:,:]); plt.title('Rec. SoS'); plt.colorbar(); plt.clim(0,1); plt.axis('off')
plt.subplot(2,4,8);plt.imshow(test_soss[n][0,0,:,:]-test_rec_soss[n][0,0,:,:], cmap='PiYG');plt.colorbar();plt.title('Rec. Error'); plt.clim(-0.15,0.15); plt.axis('off')
plt.show()
print(np.corrcoef([test_mris[n], test_rec_mris[n]]))
print(np.corrcoef([test_soss[n], test_rec_soss[n]]))

# plot difference images
# plt.subplots(2,5,figsize=(18,5))
# for i in range(1,6):
#     plt.subplot(2,5,i)
#     plt.imshow(test_mris[i-1]-test_rec_mris[i-1][0,0,:,:], cmap='PiYG');plt.colorbar();plt.title('MAE in MRIs Recons'); plt.clim(-0.5,0.5); plt.axis('off')
#     plt.subplot(2,5,i+5)
#     plt.imshow(test_soss[i-1][0,0,:,:]-test_rec_soss[i-1][0,0,:,:], cmap='PiYG');plt.colorbar();plt.title('MAE in SoS Recons'); plt.clim(-0.5,0.5); plt.axis('off')
# plt.subplots(2,5,figsize=(18,5))
# for i in range(1,6):
#     plt.subplot(2,5,i)
#     plt.imshow(test_mris[i-1]-test_fake_mris[i-1][0,0,:,:], cmap='PiYG');plt.colorbar();plt.title('MAE in MRIs'); plt.clim(-0.5,0.5); plt.axis('off')
#     plt.subplot(2,5,i+5)
#     plt.imshow(test_soss[i-1][0,0,:,:]-test_fake_soss[i-1][0,0,:,:], cmap='PiYG');plt.colorbar();plt.title('MAE in SoS'); plt.clim(-0.5,0.5); plt.axis('off')

# plt.show()

test_mris_np = []
test_soss_np = []
for i in range(len(test_mris)):
    test_mris_np.append(test_mris[i].cpu().detach().numpy())
    test_soss_np.append(test_soss[i].cpu().detach().numpy())


def average(lst):
    return sum(lst) / len(lst)

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;hgg
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA - imageB) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

mri_mses = []
sos_mses = []
for i in range(len(test_mris)):
    # error in fake MRIs
    mri_mses.append(mse(test_mris_np[i], test_fake_mris[i][0,0,:,:]))

    # error in fake SoS images
    sos_mses.append(mse(test_soss_np[i], test_fake_soss[i][0,0,:,:]))

from skimage.metrics import structural_similarity as ssim
mri_ssim = []
sos_ssim = []
for i in range(len(test_mris)):
    mri_ssim.append(ssim(test_mris_np[i], test_fake_mris[i][0,0,:,:]))
    sos_ssim.append(ssim(test_soss_np[i][0,0,:,:], test_fake_soss[i][0,0,:,:]))

def PSNR(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR has no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

mri_psnr = []
sos_psnr = []
for i in range(len(test_mris)):
    mri_psnr.append(PSNR(test_mris_np[i], test_fake_mris[i][0,0,:,:]))
    sos_psnr.append(PSNR(test_soss_np[i][0,0,:,:], test_fake_soss[i][0,0,:,:]))

import sklearn.metrics as metrics
mri_mae = []
sos_mae = []
for i in range(len(test_mris)):
    mri_mae.append(metrics.mean_absolute_error(test_mris_np[i], test_fake_mris[i][0,0,:,:]))
    sos_mae.append(metrics.mean_absolute_error(test_soss_np[i][0,0,:,:], test_fake_soss[i][0,0,:,:]))

print('Evaluation metrics:')
print('MAE: \t MRI:'+str(plt.mean(mri_mae)))
print('\t SoS: '+str(plt.mean(sos_mae)))
print('MSE: \t MRI: '+str(plt.mean(mri_mses)))
print('\t SoS: '+str(plt.mean(sos_mses)))
print('PSNR: \t MRI: '+str(plt.mean(mri_psnr)))
print('\t SoS: '+str(plt.mean(sos_psnr)))
print('SSIM: \t MRI: '+str(plt.mean(mri_ssim)))
print('\t SoS: '+str(plt.mean(sos_ssim)))



real_mris = np.mean(test_mris_np, axis=0).flatten()
real_soss = np.mean(test_soss_np[:], axis=0).flatten()
rec_soss = np.mean(np.array(test_rec_soss[:]),axis=0).flatten()
fake_soss = np.mean(np.array(test_fake_soss[:]),axis=0).flatten()
rec_mris = np.mean(np.array(test_rec_mris[:]),axis=0).flatten()
fake_mris = np.mean(np.array(test_fake_mris[:]),axis=0).flatten()

# plt.subplots(2,2, figsize=(15,15))
# plt.subplot(2,2,1)
# plt.scatter(real_mris,rec_mris)
# plt.xlabel('Original MRI', fontsize=12)
# plt.ylabel('Reconstructed MRI', fontsize=12)
# plt.subplot(2,2,2)
# plt.scatter(real_soss,rec_soss)
# plt.xlabel('Original SoS', fontsize=12)
# plt.ylabel('Reconstructed SoS', fontsize=12)
# plt.subplot(2,2,3)
# plt.scatter(real_mris, fake_mris)
# plt.xlabel('Original MRIs', fontsize=12)
# plt.ylabel('Fake MRIs', fontsize=12)
# plt.subplot(2,2,4)
# plt.scatter(real_soss,fake_soss)
# plt.xlabel('Original SoS', fontsize=12)
# plt.ylabel('Fake SoS', fontsize=12)
# # plt.show()

# plot the skull and brain graphs separately
t=n;
T = 0.6
fil_m = []
fil_s = []
fake_m = []
fake_s = []
rec_m = []
rec_s = []
m = np.array(test_mris[t]).flatten()
s = np.array(test_soss[t][0,0,:,:]).flatten()
fm = np.array(test_fake_mris[t][0,0,:,:]).flatten()
fs = np.array(test_fake_soss[t][0,0,:,:]).flatten()
rm = np.array(test_rec_mris[t][0,0,:,:]).flatten()
rs = np.array(test_rec_soss[t][0,0,:,:]).flatten()

skull_fil_m = []
skull_fil_s = []
skull_fake_m = []
skull_fake_s = []
skull_rec_m = []
skull_rec_s = []

for x in range(s.shape[0]):
    if(s[x] < T): 
        fil_m.append(m[x])
        fil_s.append(s[x])
        fake_m.append(fm[x])
        fake_s.append(fs[x])
        rec_m.append(rm[x])
        rec_s.append(rs[x])

    if(s[x] > T):
        skull_fil_m.append(m[x])
        skull_fil_s.append(s[x])
        skull_fake_m.append(fm[x])
        skull_fake_s.append(fs[x])
        skull_rec_m.append(rm[x])
        skull_rec_s.append(rs[x])
        
# plt.subplots(2,2, figsize=(8,10))
# plt.subplot(2,2,1)
# plt.scatter(skull_fil_m, skull_fake_m)
# plt.xlabel('Real MRI Values')
# plt.ylabel('Fake MRI Values')
# plt.title('Skull')
# plt.subplot(2,2,2)
# plt.scatter(skull_fil_s, skull_fake_s)
# plt.xlabel('Real SoS Values')
# plt.ylabel('Fake SoS Values')
# plt.title('Skull')
# plt.subplot(2,2,3)
# plt.scatter(skull_fil_s, skull_rec_m)
# plt.xlabel('Real MRI Values')
# plt.ylabel('Rec MRI Values')
# plt.title('Skull')
# plt.subplot(2,2,4)
# plt.scatter(skull_fil_s, skull_rec_s)
# plt.xlabel('Real SoS Values')
# plt.ylabel('Rec SoS Values')
# plt.title('Skull')

# plt.subplots(2,2, figsize=(8,10))
# plt.subplot(2,2,1)
# plt.scatter(fil_m, fake_m)
# plt.xlabel('Real MRI Values')
# plt.ylabel('Fake MRI Values')
# plt.title('Brain')
# plt.subplot(2,2,2)
# plt.scatter(fil_s, fake_s)
# plt.xlabel('Real SoS Values')
# plt.ylabel('Fake SoS Values')
# plt.title('Brain')
# plt.subplot(2,2,3)
# plt.scatter(fil_s, rec_m)
# plt.xlabel('Real MRI Values')
# plt.ylabel('Rec MRI Values')
# plt.title('Brain')
# plt.subplot(2,2,4)
# plt.scatter(fil_s, rec_s)
# plt.xlabel('Real SoS Values')
# plt.ylabel('Rec SoS Values')
# plt.title('Brain')
plt.show()
