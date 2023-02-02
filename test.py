import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case
from tqdm import tqdm
import nibabel as nib
import numpy as np
import random
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2020COVIDSeg_TrainingSet/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='vnet_ds', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/UAMT_unlabel/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [item.replace('\n', '') for item in image_list]


net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
epoch_num = 30000
save_mode_path = os.path.join(snapshot_path, 'l_model_iter_' + str(epoch_num) + '.pth')
net.load_state_dict(torch.load(save_mode_path))
print("init weight from {}".format(save_mode_path))
net.eval()
patch_size = (144,144,48)
mdc = 0
for image_path in tqdm(image_list):
    id = image_path
    image = nib.load("../data/2020COVIDSeg_TrainingSet/" + image_path + "_ct.nii/" + image_path + ".nii").get_data()
    label = nib.load("../data/2020COVIDSeg_TrainingSet/" + image_path + "_seg.nii/" + image_path + "_seg.nii").get_data()

    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape
    ws = random.randint(0,ww - patch_size[0])
    hs = random.randint(0,hh - patch_size[1])
    ds = random.randint(0,dd - patch_size[2])
    test_patch = image[ws : ws+patch_size[0] , hs:hs + patch_size[1] ,ds:ds + patch_size[2]]
    test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
    test_patch = torch.from_numpy(test_patch).cuda()
    y1 = net(test_patch)
    y = F.softmax(y1 , dim = 1)
    y = y.cpu().data.numpy()
    y = y[:,1].flatten()
    l = label[ws : ws+patch_size[0] , hs:hs + patch_size[1] ,ds:ds + patch_size[2]].flatten()
    ins = (y * l).sum() + 1e-5
    un = y.sum() + l.sum() + 1e-5
    dc = ins / un
    mdc = mdc + dc
    print(dc)
print("mdc")
print(mdc/20)

