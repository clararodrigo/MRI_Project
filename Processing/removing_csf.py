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

import os
test_mris = []
test_soss = []
ns = []
count = 0
while count<5:
    n = random.choice(os.listdir(r'/home/oab18/Projects/MRI_Project/Dataset/A')).split('.')[0].split('g')[1]
    if n not in ns:
        ns.append(n)
        test_mris.append(plt.imread(r'/home/oab18/Projects/MRI_Project/Dataset/A/mri_img'+str(n)+'.png'))
        test_soss.append(plt.imread(r'/home/oab18/Projects/MRI_Project/Dataset/B/sos_img'+str(n)+'.png'))
        count += 1
print(len(test_mris))

# plt.figure(); plt.imshow(test_soss[0])
# plt.figure(); plt.imshow(test_mris[0])
# plt.figure(); plt.scatter(test_soss[0], test_mris[0])
# plt.show()

def load_models( name):
  G_A2B=torch.load(name+'_G_A2B.pt', map_location=torch.device('cpu'))
  G_B2A=torch.load(name+"_G_B2A.pt",map_location=torch.device('cpu'))
  D_A=torch.load(name+"_D_A.pt", map_location=torch.device('cpu'))
  D_B=torch.load(name+"_D_B.pt", map_location=torch.device('cpu'))
  return G_A2B, G_B2A, D_A, D_B

G_A2B, G_B2A, D_A, D_B = load_models('full_3')


# 0.02 to 0.1
sos = test_soss.copy()
mri = test_mris.copy()
for f in range(len(sos)):
    sos[f] = sos[f].flatten()
    mri[f] = mri[f].flatten()
    for i in range(len(sos[f])):
        if(sos[f][i] < 0.02): 
            sos[f][i] = 0
            mri[f][i] = 0


for i in range(len(sos)):
    sos[i] = sos[i].reshape(256,320)
    mri[i] = mri[i].reshape(256,320)


test_fake_mris = []
test_fake_soss = []
test_rec_mris = []
test_rec_soss = []

G_A2B.to(device)
G_B2A.to(device)
D_A.to(device)
D_B.to(device)
for i in range(len(mri)):
    real_mri = torch.Tensor(mri[i]).unsqueeze(0).unsqueeze(0)
    fake_sos = G_A2B(real_mri.to(device)).cpu().detach().numpy()
    rec_mri = G_B2A(torch.Tensor(fake_sos).to(device)).cpu().detach().numpy()

    real_sos = torch.Tensor(sos[i]).unsqueeze(0).unsqueeze(0)
    fake_mri = G_B2A(real_sos.to(device)).cpu().detach().numpy()
    rec_sos = G_A2B(torch.Tensor(fake_mri).to(device)).cpu().detach().numpy()

    test_fake_mris.append(fake_mri)
    test_fake_soss.append(fake_sos)
    test_rec_mris.append(rec_mri)
    test_rec_soss.append(rec_sos)

    with torch.no_grad():
        real_mri.detach().cpu()
        real_sos.detach().cpu()


plt.figure()
plt.imshow(test_fake_mris[0][0,0,:,:]-test_soss[0],'PiYG')
plt.colorbar()
plt.clim(-1,1)
plt.show()

plt.figure()
plt.imshow(test_fake_soss[0][0,0,:,:]-test_mris[0],'PiYG')
plt.colorbar()
plt.clim(-1,1)
plt.show()

c=0
plt.subplots(3,2)
for i in range(1,7,2):
    plt.subplot(3,2,i)
    plt.imshow(mri[c])
    plt.title('Real MRI, no CSF')
    plt.colorbar()
    plt.clim(0,1)
    plt.subplot(3,2,i+1)
    plt.imshow(test_fake_mris[c][0,0,:,:])
    plt.title('Fake MRI, no CSF')
    plt.colorbar()
    plt.clim(0,1)
    c+=1
plt.show()

c=0
plt.subplots(3,2)
for i in range(1,7,2):
    plt.subplot(3,2,i)
    plt.imshow(sos[c])
    plt.title('Real SoS, no CSF')
    plt.colorbar()
    plt.clim(0,1)
    plt.subplot(3,2,i+1)
    plt.imshow(test_fake_soss[c][0,0,:,:])
    plt.title('Fake SoS, no CSF')
    plt.colorbar()
    plt.clim(0,1)
    c+=1
plt.show()

def average(lst):
    return sum(lst) / len(lst)

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

mri_mses = []
sos_mses = []
for i in range(len(test_mris)):
    # error in fake MRIs
    mri_mses.append(mse(test_mris[i], test_fake_mris[i][0,0,:,:]))

    # error in fake SoS images
    sos_mses.append(mse(test_soss[i], test_fake_soss[i][0,0,:,:]))

from skimage.metrics import structural_similarity as ssim
mri_ssim = []
sos_ssim = []
for i in range(len(test_mris)):
    mri_ssim.append(ssim(test_mris[i], test_fake_mris[i][0,0,:,:]))
    sos_ssim.append(ssim(test_soss[i], test_fake_soss[i][0,0,:,:]))

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
    mri_psnr.append(PSNR(test_mris[i], test_fake_mris[i][0,0,:,:]))
    sos_psnr.append(PSNR(test_soss[i], test_fake_soss[i][0,0,:,:]))

import sklearn.metrics as metrics
mri_mae = []
sos_mae = []
for i in range(len(test_mris)):
    mri_mae.append(metrics.mean_absolute_error(test_mris[i], test_fake_mris[i][0,0,:,:]))
    sos_mae.append(metrics.mean_absolute_error(test_soss[i], test_fake_soss[i][0,0,:,:]))

print('Evaluation metrics:')
print('MAE: \t MRI:'+str(plt.mean(mri_mae)))
print('\t SoS: '+str(plt.mean(sos_mae)))
print('MSE: \t MRI: '+str(plt.mean(mri_mses)))
print('\t SoS: '+str(plt.mean(sos_mses)))
print('PSNR: \t MRI: '+str(plt.mean(mri_psnr)))
print('\t SoS: '+str(plt.mean(sos_psnr)))
print('SSIM: \t MRI: '+str(plt.mean(mri_ssim)))
print('\t SoS: '+str(plt.mean(sos_ssim)))


real_mris = np.mean(np.array(test_mris[:]),axis=0).flatten()
real_soss = np.mean(np.array(test_soss[:]),axis=0).flatten()
rec_soss = np.mean(np.array(test_rec_soss[:]),axis=0).flatten()
fake_soss = np.mean(np.array(test_fake_soss[:]),axis=0).flatten()
rec_mris = np.mean(np.array(test_rec_mris[:]),axis=0).flatten()
fake_mris = np.mean(np.array(test_fake_mris[:]),axis=0).flatten()

# plot the skull and brain graphs separately
t=1;
T = 0.37
fil_m = []
fil_s = []
fake_m = []
fake_s = []
rec_m = []
rec_s = []
m = np.array(mri[t]).flatten()
s = np.array(sos[t]).flatten()
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
# plt.show()

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
# plt.show()
