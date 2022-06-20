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
    plt.imshow(sos_data[0])
    plt.title('SoS before processing')
    plt.colorbar(); 
    plt.clim(0,3000)
    plt.subplot(2,2,2)
    plt.imshow(sos[0])
    plt.title('SoS after processing')
    plt.colorbar(); 
    # plt.clim(0,3000)
    plt.subplot(2,2,3)
    plt.imshow(mri_data[0])
    plt.title('MRI before processing')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(mri[0])
    plt.title('MRI after processing')
    plt.colorbar()
    plt.show()

def make_ellipse(parameters, x, y):
    c_x = parameters[0]
    c_y = parameters[1]
    r_x = parameters[2]
    r_y = parameters[3]
    
    a = (x - c_x)**2
    b = (r_x**2)
    c = (y - c_y)**2
    d = (r_y**2)
    
    ellipse = (a/b) + (c/d)
    # print(ellipse)
    return ellipse

mri_files = glob(r'/run/media/oab18/f2b6f79d-ec9b-4538-a442-90e74fb156bf/head-datasets/Ultrasound-MRI_volumes/*/m*_T1w.npy.gz')
sos_files = glob(r'/run/media/oab18/f2b6f79d-ec9b-4538-a442-90e74fb156bf/head-datasets/Ultrasound-MRI_volumes/*/*Vp.npy.gz')

print("Number of MRIs: "+str(len(mri_files)))
mri_data = []
sos_data = []
slice_list = []
for i in range(1): # just using 150 files for train and from 995
    print(i)
    mri_temp = gzip.GzipFile(mri_files[i],'r') 
    sos_temp = gzip.GzipFile(sos_files[i],'r')
    mri_slice = np.load(mri_temp)
    sos_slice = np.load(sos_temp)

    for s in range(150,230):#,230):
        mri_data.append(mri_slice[:,:,s])
        sos_data.append(sos_slice[:,:,s])
        slice_list.append(s)
print('Total slices: '+str(len(mri_data)))

# mean_mri_img = np.load('mean_mri_head.npy')
# mean_sos_img = np.load('mean_sos_head.npy')
# stdev_mri_img = np.load('stdev_mri_head.npy')
# stdev_sos_img = np.load('stdev_sos_head.npy')
c_x = 150
c_y = 125
r_x = 120
r_y = 70
for i in range(len(mri_data)):
    for x in range(len(mri_data[i][:,0])):
        for y in range(len(mri_data[i][0,:])):
            if sos_data[i][x,y] < 1600: mri_data[i][x,y] = 0
            if((make_ellipse([c_y,c_x,r_y,r_x],x,y)<=1) ):
                mri_data[i][x,y] = 0
            if((make_ellipse([127,152,110,150],x,y)>1) and mri_data[i][x,y] > 800):
                mri_data[i][x,y] = 0
            if((make_ellipse([127,235,70,35],x,y)<1) and mri_data[i][x,y] > 800):
                mri_data[i][x,y] = 0
plt.figure(); plt.imshow(mri_data[0]); plt.show()
prob = 0.04
thres = 1 - prob
stdev_noise = 75
sos = sos_data.copy()
mri = mri_data.copy()
for f in range(len(mri)):
    print(f)
    sos[f] = sos[f].flatten()
    for x in range(len(mri[f][:,0])):
        for y in range(len(mri[f][0,:])):
            rdn = random.randint(-1,1)
            sos[f][i] = sos[f][i]+rdn*stdev_noise
            mri[f][x,y] = mri[f][x,y]+rdn*stdev_noise
print('Data clipped!')



print(np.max(mri))
print(np.max(sos))
mri = mri/np.max(np.max(mri))
# for i in range(30):
#     plt.figure(); plt.imshow(mri[i]); 
# plt.show()

# plt.scatter(mri[0],sos[0]); plt.show()
 
# torch.save(torch.Tensor(mri),r'/home/oab18/Projects/MRI_Project/Dataset/ct_skulls/test/mri_dataset.pt')
# torch.save(torch.Tensor(sos),r'/home/oab18/Projects/MRI_Project/Dataset/ct_skulls/train/sos_dataset.pt')


# for i in range(len(mri)):
#     imageio.imwrite(r'/home/oab18/Projects/MRI_Project/Dataset/noisy/A/mri_img'+str(i)+'.png', mri[i])
#     imageio.imwrite(r'/home/oab18/Projects/MRI_Project/Dataset/noisy/B/sos_img'+str(i)+'.png', sos[i])
