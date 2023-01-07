# -*-codeing = utf-8 -*-
# @Time : 2022/9/16 16:02
# @Author : LichunSun
# @File : test.py

from astropy.io import fits
import cv2

import numpy as np
img = fits.open('/home/dell460/slc/sdd_01/Faster_mul/traindata300-300/image_random_2048_5120.fits')[0].data
#img = fits.open('/home/dell460/slc/sdd_01/Faster_mul/traindata300-300_t1t2/image_random_2048_5120.fits')[0].data

print(img.shape)
# for i in range(img.shape[0]):
#     #print(np.max(img[i,::]))
#     if np.max(img[i,::])==0:
#         print('s')
im_scale = 0.56
# for i in range(img.shape[0]):
#     im = cv2.resize(img[i,::], None, None, fx=im_scale, fy=im_scale,
#                     interpolation=cv2.INTER_LINEAR)
new_img = img[96*5:96*6,::]
# a = np.copy(img)
# print(a)
a = np.empty((1024,1024,513))
print(a.shape[2])
# im = new_img.astype(np.float32, copy=False)
# if len(im.shape) == 3:
#     im = np.swapaxes(im, 0, 1)  # 8,256,256->256,8,256
#     im = np.swapaxes(im, 1, 2)
# # im -= pixel_means
# max_value = np.max(im)
# min_value = np.min(im)
# mean_value = np.mean(im)
# im = (im - mean_value) / (max_value - min_value)
# #im_scale = float(target_size) / float(im_size_min)
# print(im.shape)
print(a[:,:,:5].shape)
im = cv2.resize(a[:,:,:513], None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

