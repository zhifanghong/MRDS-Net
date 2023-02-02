import numpy
import torch
import numpy as np
import nibabel as nib
import h5py
import os
import argparse
import dicom2nifti
from utils import ramps
import cv2 as cv
from pydicom import dicomio
from code1.networks.vnet import VNet
import scipy
from tqdm import tqdm
#0.859375, 0.859375
# 2
###############预处理

# def resample(image, new_spacing=[1,1,1]):
#     # Determine current pixel spacing
#     spacing = np.array([0.857375 , 0.859375 , 2])
#
#     resize_factor = spacing / new_spacing
#     new_real_shape = image.shape * resize_factor
#     new_shape = np.round(new_real_shape)
#     real_resize_factor = new_shape / image.shape
#     new_spacing = spacing / real_resize_factor
#
#     image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
#
#     return image, new_spacing
#

#NIH图像预处理
# fname = os.listdir("G:\DATASET/NIH/NII\image")
# for f in tqdm(fname):
#     image_nii = nib.load("G:\DATASET/NIH/NII\image/" + f)
#     label_nii = nib.load("G:\DATASET/NIH/NII\label/" + f)
#     image = image_nii.get_data()
#     label = label_nii.get_data()
#     #new_image,new_image_spacing = resample(image)
#     #new_label,new_label_spacing = resample(label)
#     #new_image = new_image.clip(-100 , 240) + 100
#
#     #使用-100 和 240阈值对hu值进行截断
#     new_image = image.clip(-100 , 240) + 100
#
#     #提取目标区域
#     tempIMG = np.nonzero(new_image)
#     minx, maxx = np.min(tempIMG[0]), np.max(tempIMG[0])
#     miny, maxy = np.min(tempIMG[1]), np.max(tempIMG[1])
#     minz, maxz = np.min(tempIMG[2]), np.max(tempIMG[2])
#     new_image = new_image[minx:maxx + 1 , miny:maxy + 1 , minz:maxz + 1] - 100
#     new_label = label[minx:maxx + 1 , miny:maxy + 1 , minz:maxz + 1]
#
#     #使用包含胰腺切片以及胰腺上下20张的切片
#     tempL = np.nonzero(label)
#     z_min ,z_max = np.min(tempL[2]) , np.max(tempL[2])
#     new_image = new_image[:,:,max(z_min - 20 , 0) :min( z_max + 21 , maxz + 1) ]
#     new_label = new_label[:,:,max(z_min - 20 , 0) :min( z_max + 21 , maxz + 1) ]
#
#     save_image = nib.Nifti1Image(new_image , image_nii.affine)
#     save_label = nib.Nifti1Image(new_label , label_nii.affine)
#     nib.save(save_image , "G:\DATASET/NIH/NII\image_process/" + f)
#     nib.save(save_label , "G:\DATASET/NIH/NII\label_process/" + f)
#
#     ff = h5py.File("G:\DATASET/NIH/NII\h5_process/" + f.replace(".nii" , "_h5.h5"), 'w')
#     ff.create_dataset('image', data=new_image, compression="gzip")
#     ff.create_dataset('label', data=new_label, compression="gzip")
#     ff.close()
#NII训练图像均值0.17210566467820265


#LITS预处理
# fname = os.listdir("G:\DATASET\LITS2017\LITS/image/")
# for f in tqdm(fname):
#     image_nii = nib.load("G:\DATASET\LITS2017\LITS/image/" + f)
#     label_nii = nib.load("G:\DATASET\LITS2017\LITS/label/" + f)
#     image = image_nii.get_data()
#     label = label_nii.get_data()
#     w, h, d = label.shape
#     #使用包含肝脏切片以及肝脏上下20张的切片
#     tempL = np.nonzero(label)
#     z_min, z_max = np.min(tempL[2]), np.max(tempL[2])
#     image = image[:,:,max(z_min - 20 , 0) :min( z_max + 21 , image.shape[2] + 1)]
#     label = label[:,:,max(z_min - 20 , 0) :min( z_max + 21 , label.shape[2] + 1)]
#     print(image.shape)
#     print(label.shape)
#     #以-200和200两个阈值截取hu值
#     image = image.clip(-200 , 200)
#
#     #提取目标区域
#     image = image + 200
#     tempIMG = np.nonzero(image)
#     minx, maxx = np.min(tempIMG[0]), np.max(tempIMG[0])
#     miny, maxy = np.min(tempIMG[1]), np.max(tempIMG[1])
#     minz, maxz = np.min(tempIMG[2]), np.max(tempIMG[2])
#     new_image = image[minx:maxx + 1 , miny:maxy + 1 , minz:maxz + 1] - 200
#     new_label = label[minx:maxx + 1 , miny:maxy + 1 , minz:maxz + 1]
#     print(image.shape)
#     print(label.shape)
#     #保存image和label
#     save_image = nib.Nifti1Image(new_image , image_nii.affine)
#     save_label = nib.Nifti1Image(new_label , label_nii.affine)
#     nib.save(save_image , "G:\DATASET\LITS2017\LITS/train\image/" + f)
#     nib.save(save_label , "G:\DATASET\LITS2017\LITS/train\label/" + f)
#
#     ff = h5py.File("G:\DATASET\LITS2017\LITS/train\h5/" + f.replace(".nii" , "_h5.h5"), 'w')
#     ff.create_dataset('image', data=new_image, compression="gzip")
#     ff.create_dataset('label', data=new_label, compression="gzip")
#     ff.close()

#COVID预处理

#无标签数据处理
print("有标签数据处理")
unlabel_fname = os.listdir("G:\DATASET/temp_unlabeled_h5")
for f in tqdm(unlabel_fname):
    print(f)
    image_nii = nib.load("G:\DATASET/temp_unlabeled_h5/" + f)
    image = image_nii.get_data()

    #以-500和500两个阈值截取hu值
    image = image.clip(-1000 , 400)

    #提取目标区域
    image = image + 1000
    tempIMG = np.nonzero(image)
    minx, maxx = np.min(tempIMG[0]), np.max(tempIMG[0])
    miny, maxy = np.min(tempIMG[1]), np.max(tempIMG[1])
    minz, maxz = np.min(tempIMG[2]), np.max(tempIMG[2])
    new_image = image[minx:maxx + 1 , miny:maxy + 1 , minz:maxz + 1] - 1000
    label = np.zeros(image.shape , dtype= np.float32)
    #保存image和label
    save_image = nib.Nifti1Image(new_image , image_nii.affine)
    nib.save(save_image , "G:\DATASET/COVID19/unlabeled_image/" + f)

    ff = h5py.File("G:\DATASET\COVID19\h5/" + f.replace(".nii" , "_h5.h5"), 'w')
    ff.create_dataset('image', data=new_image, compression="gzip")
    ff.create_dataset('label', data=label, compression="gzip")
    ff.close()


fname = os.listdir("G:\DATASET\Challenge19-20/")
#有标签数据处理
print("有标签数据处理")
for f in tqdm(fname):
    if "seg" in f:
        continue
    print(f)
    image_nii = nib.load("G:\DATASET\Challenge19-20/" + f + '/' + f.replace("_ct" , ""))
    label_nii = nib.load("G:\DATASET\Challenge19-20/" + f.replace("ct" , "seg") + '/' +f.replace("ct" , "seg"))
    image = image_nii.get_data()
    label = label_nii.get_data()
    w, h, d = label.shape
    #使用包含新冠切片以及新冠上下20张的切片
    tempL = np.nonzero(label)
    z_min, z_max = np.min(tempL[2]), np.max(tempL[2])
    image = image[:,:,max(z_min - 20 , 0) :min( z_max + 21 , image.shape[2] + 1)]
    label = label[:,:,max(z_min - 20 , 0) :min( z_max + 21 , label.shape[2] + 1)]
    print(image.shape)
    print(label.shape)
    #以-500和500两个阈值截取hu值
    image = image.clip(-1000 , 400)

    #提取目标区域
    image = image + 1000
    tempIMG = np.nonzero(image)
    minx, maxx = np.min(tempIMG[0]), np.max(tempIMG[0])
    miny, maxy = np.min(tempIMG[1]), np.max(tempIMG[1])
    minz, maxz = np.min(tempIMG[2]), np.max(tempIMG[2])
    new_image = image[minx:maxx + 1 , miny:maxy + 1 , minz:maxz + 1] - 1000
    new_label = label[minx:maxx + 1 , miny:maxy + 1 , minz:maxz + 1]
    print(image.shape)
    print(label.shape)
    #保存image和label
    save_image = nib.Nifti1Image(new_image , image_nii.affine)
    save_label = nib.Nifti1Image(new_label , label_nii.affine)
    nib.save(save_image , "G:\DATASET\COVID19\labeled_image/" + f)
    nib.save(save_label , "G:\DATASET\COVID19\labeled_label/" + f)

    ff = h5py.File("G:\DATASET\COVID19\h5/" + f.replace("_ct.nii" , "_h5.h5"), 'w')
    ff.create_dataset('image', data=new_image, compression="gzip")
    ff.create_dataset('label', data=new_label, compression="gzip")
    ff.close()











