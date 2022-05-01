import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pylab as plt

import glob
from glob import glob
import nibabel as nb
import os
import pydicom as dicom

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed_all(42)
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

path = '/run/media/oab18/f2b6f79d-ec9b-4538-a442-90e74fb156bf/head-datasets/RSNA_25k/volumes/*/*/*'
files = glob(path)
print(len(files))

images = []
for i in range(-2000,0):
    img = glob(files[i])
    for j in range(len(img)):
        ds = dicom.dcmread(img[j])
        a = ds.pixel_array
        if(np.mean(a) > 400):
            images.append(a)
            print('.',end='')
    

# add noise
stdev_noise = 85
ct = np.array(images)
flat_ct = []
c_x = 230
c_y = 260
r_x = 180
r_y = 220
for f in range(len(ct)):
    print(f)
    
    for x in range(len(ct[f][:,0])): 
        for y in range(len(ct[f][0,:])): 

            rdn = random.randint(-1,1)
            ct[f][x][y] = ct[f][x][y] + rdn*stdev_noise
            
            if((make_ellipse([c_y,c_x,r_y,r_x], x, y) > 1) and (ct[f][x][y] >300)):
                ct[f][x][y] = 0


images = torch.from_numpy(np.int16(ct));
images = torchvision.transforms.functional.rotate(images,270)

images = torchvision.transforms.Pad((64,0,64,0))(images)
p = torchvision.transforms.Compose([transforms.Resize((256,320))])
smol_images = []
for i in range(len(images)):
    smol_images.append(p(images[i].unsqueeze(0).unsqueeze(0)))



torch.save(smol_images,r'/home/oab18/Projects/MRI_Project/Dataset/mri2ct2sos_dataset/test/ct_dataset.pt')
