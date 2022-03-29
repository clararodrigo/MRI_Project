import SimpleITK as sitk
import os
import re
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import gzip
import glob
from glob import glob

path_out = '/home/oab18/Projects/MRI_Project/'
sos_path = glob(r'/run/media/oab18/f2b6f79d-ec9b-4538-a442-90e74fb156bf/head-datasets/Ultrasound-MRI_volumes/*/Vp.npy.gz')
mri_path = glob(r'/run/media/oab18/f2b6f79d-ec9b-4538-a442-90e74fb156bf/head-datasets/Ultrasound-MRI_volumes/*/m*_T1w.npy.gz')

# # calculate average images
# mean_mri_img = np.zeros((256,320,320))
# mean_sos_img = np.zeros((256,320,320))
# n = 1
# for i in range(len(mri_path)):
#     print(n)
#     try:
#         mri_img = gzip.GzipFile(mri_path[i],'r') 
#         mri_img = np.load(mri_img)

#         mean_mri_img = (mean_mri_img*(n-1)+mri_img)/n

#         sos_img = gzip.GzipFile(sos_path[i],'r') 
#         sos_img = np.load(sos_img)

#         mean_sos_img = (mean_sos_img*(n-1)+sos_img)/n
#         n+=1
#     except:
#         continue
    


# # save average images
# np.save(path_out+'mean_mri_head', mean_mri_img)
# np.save(path_out+'mean_sos_head', mean_sos_img)

# mean_mri_img = np.load(path_out+'mean_mri_head.npy')
# mean_sos_img = np.load(path_out+'mean_sos_head.npy')

# calculate stdev of images
# stdev_mri_img = np.zeros((256,320,320))
# stdev_sos_img = np.zeros((256,320,320))
# n = 1
# for i in range(len(mri_path)):
#     print(n)
#     try:
#         mri_img = gzip.GzipFile(mri_path[i],'r') 
#         mri_img = np.load(mri_img)
#         stdev_mri_img = np.sqrt((stdev_mri_img*(n-1)+(mri_img-mean_mri_img)**2)/n)

#         sos_img = gzip.GzipFile(sos_path[i],'r') 
#         sos_img = np.load(sos_img)
#         stdev_sos_img = np.sqrt((stdev_sos_img*(n-1)+(sos_img-mean_sos_img)**2)/n)
#         n+=1
#     except:
#         continue

# np.save(path_out+'stdev_mri_head', stdev_mri_img)
# np.save(path_out+'stdev_sos_head', stdev_sos_img)

# correcting var to stdev
# stdev_mri_img = np.load(path_out+'stdev_mri_head.npy')
# stdev_sos_img = np.load(path_out+'stdev_sos_head.npy')
# for l in range(len(stdev_mri_img[:,:,0])):
#     stdev_mri_img[:,:,l] = np.sqrt(stdev_mri_img[:,:,l])
#     stdev_sos_img[:,:,l] = np.sqrt(stdev_sos_img[:,:,l])

# np.save(path_out+'stdev_mri_head', stdev_mri_img)
# np.save(path_out+'stdev_sos_head', stdev_sos_img)

mean_mri_img = np.load(path_out+'mean_mri_head.npy')
stdev_mri_img = np.load(path_out+'stdev_mri_head.npy')
mean_sos_img = np.load(path_out+'mean_sos_head.npy')
stdev_sos_img = np.load(path_out+'stdev_sos_head.npy')

dims = mean_mri_img.shape
# # plot mean MRIs in different axes
# plt.figure()
# plt.imshow(np.flipud( mean_mri_img[:,:,int(dims[2]/2)] ))
# plt.colorbar()
# plt.clim(0,1000)
# plt.title('Mean, dim 2')
# plt.figure()
# plt.imshow(np.flipud( mean_mri_img[:,int(dims[1]/2),:] ))
# plt.title('Mean, dim 1')
# plt.clim(0,1000)
# plt.figure()
# plt.imshow(np.flipud( mean_mri_img[int(dims[0]/2),:,:] )) # 40
# plt.title('Mean, dim 0')
# plt.clim(0,1000)

# # plot stdev MRIs in different axes
# plt.figure()
# plt.imshow(np.flipud( stdev_mri_img[:,:,int(dims[2]/2)] ))
# plt.colorbar()
# plt.clim(0,1000)
# plt.title('Std dev, dim 2')
# plt.figure()
# plt.imshow(np.flipud( stdev_mri_img[:,int(dims[1]/2),:] ))
# plt.clim(0,1000)
# plt.title('Std dev, dim 1')
# plt.figure()
# plt.imshow(np.flipud( stdev_mri_img[int(dims[0]/2),:,:] ))
# plt.clim(0,1000)
# plt.title('Std dev, dim 0')

# show max and min MRI)
max_mri = mean_mri_img + 3*stdev_mri_img
min_mri = mean_mri_img - 3*stdev_mri_img
max_sos = mean_sos_img + 3*stdev_sos_img
min_sos = mean_sos_img - 3*stdev_sos_img

# plt.figure()
# plt.imshow(np.flipud( mean_mri_img[:,:,int(dims[2]/2)] ) + 3*np.flipud( stdev_mri_img[:,:,int(dims[2]/2)] ))
# plt.colorbar()
# # plt.clim(0,1000)
# plt.title('Max MRI')
# plt.figure()
# plt.imshow(np.flipud( mean_mri_img[:,:,int(dims[2]/2)] ) - 3*np.flipud( stdev_mri_img[:,:,int(dims[2]/2)] ))
# plt.colorbar()
# # plt.clim(0,1000)
# plt.title('Min MRI')

# # show max and min SoS
# plt.figure()
# plt.imshow(np.flipud( mean_sos_img[:,:,int(dims[2]/2)] ) + 3*np.flipud( stdev_sos_img[:,:,int(dims[2]/2)] ))
# plt.colorbar()
# plt.clim(0,3000)
# plt.title('Max SoS')
# plt.figure()
# plt.imshow(np.flipud( mean_sos_img[:,:,int(dims[2]/2)] ) - 3*np.flipud( stdev_sos_img[:,:,int(dims[2]/2)] ))
# plt.colorbar()
# plt.clim(0,3000)
# plt.title('Min SoS')

# see difference
plt.figure();plt.imshow((max_mri-min_mri)[:,:,160],cmap='PiYG');plt.colorbar();plt.title("Difference in MRI")
plt.figure();plt.imshow((max_sos-min_sos)[:,:,160],cmap='PiYG');plt.colorbar();plt.title("Difference in SoS")

# np.save('max_mri.npy', )

# # visualize how much stdev there is in differnt areas of the images
# plt.figure();plt.hist(stdev_sos_img[:,:,int(dims[2]/2)]); plt.title('SoS stdev histogram')
# plt.figure();plt.hist(stdev_mri_img[:,:,int(dims[2]/2)]); plt.title("MRI stdev histogram")
# plt.figure();plt.imshow(stdev_mri_img[:,:,int(dims[2]/2)]); plt.title('Stdev MRI'); plt.colorbar()
# plt.figure();plt.imshow(stdev_sos_img[:,:,int(dims[2]/2)]); plt.title('STdev SoS'); plt.colorbar()

# # visualize mean in differnt areas of the images
# plt.figure();plt.hist(mean_sos_img[:,:,int(dims[2]/2)]); plt.title('SoS mean histogram')
# plt.figure();plt.hist(mean_mri_img[:,:,int(dims[2]/2)]); plt.title("MRI mean histogram")
# plt.figure();plt.imshow(mean_mri_img[:,:,int(dims[2]/2)]); plt.title('mean MRI'); plt.colorbar()
# plt.figure();plt.imshow(mean_sos_img[:,:,int(dims[2]/2)]); plt.title('mean SoS'); plt.colorbar()


plt.show()
