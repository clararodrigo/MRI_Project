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

import gzip
import os

import cv2
import imageio
import zarr

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = image[i][j]-200
            elif rdn > thres:
                output[i][j] = image[i][j]+200
    return output

def plot():
    plt.subplots(2,2)
    plt.subplot(2,2,1)
    plt.imshow(sos_data[51])
    plt.title('SoS before processing')
    plt.colorbar(); plt.clim(1500,3000)
    plt.subplot(2,2,2)
    plt.imshow(sos[51])
    plt.title('SoS after processing')
    plt.colorbar(); plt.clim(1500,3000)
    plt.subplot(2,2,3)
    plt.imshow(mri_data[51])
    plt.title('MRI before processing')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(mri[51])
    plt.title('MRI after processing')
    plt.colorbar()
    plt.show()

mri_files = glob(r'/run/media/oab18/f2b6f79d-ec9b-4538-a442-90e74fb156bf/head-datasets/Ultrasound-MRI_volumes/*/m*_T1w.npy.gz')
sos_files = glob(r'/run/media/oab18/f2b6f79d-ec9b-4538-a442-90e74fb156bf/head-datasets/Ultrasound-MRI_volumes/*/*Vp.npy.gz')

print("Number of MRIs: "+str(len(mri_files)))
mri_data = []
sos_data = []
slice_list = []
for i in range(0,100): # just using 50 files will already give us 4000 images
    print(i)
    mri_temp = gzip.GzipFile(mri_files[i],'r') 
    sos_temp = gzip.GzipFile(sos_files[i],'r')
    mri_slice = np.load(mri_temp)
    sos_slice = np.load(sos_temp)

    for s in range(150,230):
        mri_data.append(mri_slice[:,:,s])
        sos_data.append(sos_slice[:,:,s])
        slice_list.append(s)
print('Total slices: '+str(len(mri_data)))

mean_mri_img = np.load('mean_mri_head.npy')
mean_sos_img = np.load('mean_sos_head.npy')
stdev_mri_img = np.load('stdev_mri_head.npy')
stdev_sos_img = np.load('stdev_sos_head.npy')

max_mri = mean_mri_img + 3*stdev_mri_img
min_mri = mean_mri_img - 3*stdev_mri_img
max_sos = mean_sos_img + 3*stdev_sos_img
min_sos = mean_sos_img - 3*stdev_sos_img

prob = 0.04
thres = 1 - prob
sos = sos_data.copy()
mri = mri_data.copy()
mask = [[0 for i in range(81920)] for i in range(len(sos))]
for f in range(len(sos)):
    print(f)
    sos[f] = sos[f].flatten()
    mri[f] = mri[f].flatten()
    max_sos_temp = max_sos[:,:,slice_list[f]].flatten()
    min_sos_temp = min_sos[:,:,slice_list[f]].flatten()
    max_mri_temp = max_mri[:,:,slice_list[f]].flatten()
    min_mri_temp = min_mri[:,:,slice_list[f]].flatten()

    for i in range(len(sos[f])): 
        # if(sos[f][i] > max_sos_temp[i]): # clip maxes of sos
        #     sos[f][i] = max_sos_temp[i]
        # if(sos[f][i] < min_sos_temp[i]): # clip mins of sos
        #     sos[f][i] = min_sos_temp[i]
        # if(mri[f][i] > max_mri_temp[i]): # clip maxes of mri
        #     mri[f][i] = max_mri_temp[i]
        # if(mri[f][i] < min_mri_temp[i]): # clip mins of mri
        #     mri[f][i] = min_mri_temp[i]
        if(1480< sos[f][i] <= 1540): # to add noise to CSF
            rdn = random.random()
            if(rdn < prob):
                sos[f][i] = sos[f][i]-200
            if(rdn > thres):
                sos[f][i] = sos[f][i]+200
            if(rdn < prob):
                mri[f][i] = mri[f][i]-200
            if(rdn > thres):
                mri[f][i] = mri[f][i]+200
        if(sos[f][i] <= 1480): # to filter background later
            sos[f][i] = 0
            mri[f][i] = 0
print('Data filtered!')
print('Data clipped!')


for i in range(len(sos)):
    sos[i] = sos[i].reshape(256,320)
    mri[i] = mri[i].reshape(256,320)
# plot()

# # add s&p noise
# for i in range(len(sos)):
#     sos[i] = sp_noise(sos[i],0.02)
#     mri[i] = sp_noise(mri[i],0.02)
# print('Data seasoned!')
# plot()

torch.save(torch.Tensor(mri),r'/home/oab18/Projects/MRI_Project/Dataset/noisy_2/train/mri_dataset.pt')
torch.save(torch.Tensor(sos),r'/home/oab18/Projects/MRI_Project/Dataset/noisy_2/train/sos_dataset.pt')

# for i in range(len(mri)):
    # imageio.imwrite(r'/home/oab18/Projects/MRI_Project/Dataset/clipped_noisy/A/mri_img'+str(i)+'.png', mri[i])
    # imageio.imwrite(r'/home/oab18/Projects/MRI_Project/Dataset/clipped_noisy/B/sos_img'+str(i)+'.png', sos[i])