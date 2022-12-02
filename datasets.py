import random
import os
import re

import torch
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from preprocess import *
from utils import *

###################################################
# Cyclegan
###################################################
# train dataset of cyclegan
class ImageDataset(Dataset):
    '''
    Dataloader of cyclegan
    '''
    def __init__(self, transforms_):
        self.transform = transforms_
        self.img_A = nii_loader(str='T2',is_label=False)
        self.img_B = nii_loader(str='LGE',is_label=False)
        self.len_a = len(self.img_A)
        self.len_b = len(self.img_B)

    def __getitem__(self, index):
        # warning: make sure index does't out of range
        index_a = index % self.len_a
        index_b = index % self.len_b 
        # apply torch transform
        a_tensor = torch.from_numpy(self.img_A[index_a]).float().unsqueeze(0)
        item_A = self.transform(a_tensor)
        # minmaxscaler and normalization
        item_A = min_max(item_A)
        # random load B domain image
        b_tensor = torch.from_numpy(self.img_B[random.randint(0, self.len_b - 1)]).float().unsqueeze(0)
        item_B = self.transform(b_tensor)
        item_B = min_max(item_B)
        
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(self.len_a, self.len_b)

def nii_reader(fileroot,is_label = False, do_resmaple = True):
    '''
    Method: 
        read one .nii.gz or .nii and return all slices
    
    Parameters:
        `fileroot` -- nii file relative root
        `is_label` -- is label or not
        `do_resample` -- do resmaple or not
    Return:
        re_slice: 
            all slices of given file , dtype is numpy.float32 list, size is [N, H, W]
            original range( > 255.0)
    '''

    re_slice = []
    volumns = sitk.ReadImage(fileroot)
    if do_resmaple:
        volumns = resample_image(volumns,is_label=is_label)
    # convert to float32
    data = sitk.GetArrayFromImage(volumns)
    data = data.astype(np.float32)
    for i in range(data.shape[0]):
        img_ = data[i]
        re_slice.append(img_)
    return re_slice

def nii_loader(str, is_label=False):
    '''
    Method:
        load all patient images or lables by given str
        str includes:
            all : all images or labels
            C0 : all bSSFP images or labels
            T2 : all T2-weight images or labels
            LGE : all LGE CMR or labels
    
    Parameters:
        `str` -- choosed type
        `is_label` -- load label or not
    Return :
        a list of image [h,w]
    '''
    
    dataroot = 'datasets/train/all_image'
    if is_label:
        dataroot = 'datasets/train/all_label'
    filename = os.listdir(dataroot)
    filename.sort(key=lambda x:int(x.split('_')[0][7:]))
    # select filename by str
    if str != 'all':
        filename = list(filter(lambda x: re.search(str,x) is not None,filename))
    re_images = []
    # get image and info respectively
    for name in filename:
        # get given file all slices
        fileroot = os.path.join(dataroot,name)
        assert os.path.exists(fileroot)
        images = nii_reader(fileroot,is_label,do_resmaple=True)
        re_images.extend(images)
    
    return re_images

###############################################
# Segmentation
###############################################

def getfilepath(str_):
    '''
    get all file path
    Input:
        `str_` -- image type
    Output:
        a dict {'image':[path],'label':[path]}
    '''
    allimageroot = 'datasets/train/all_image/'
    alllabelroot = 'datasets/train/all_label/'
    c0lgeroot = 'datasets/train/c0_lge/'
    t2lgeroot = 'datasets/train/t2_lge/'
    re = {'image':[],'label':[]}
    def generate(number,suffix,root):
        paths = []
        for i in range(1,number+1):
            path = root +'patient'+str(i) + suffix
            paths.append(path)
        return paths
    if str_ == 'C0LGE':
        re['image'] = generate(35,'_C0.nii.gz',c0lgeroot)
        re['label'] = generate(35,'_C0_manual.nii.gz',alllabelroot)
    if str_ == 'T2LGE':
        re['image'] = generate(35,'_T2.nii.gz',t2lgeroot)
        re['label'] = generate(35,'_T2_manual.nii.gz',alllabelroot)
    if str_ == 'C0':
        re['image'] = generate(35,'_C0.nii.gz',allimageroot)
        re['label'] = generate(35,'_C0_manual.nii.gz',alllabelroot)
    if str_ == 'T2':
        re['image'] = generate(35,'_T2.nii.gz',allimageroot)
        re['label'] = generate(35,'_T2_manual.nii.gz',alllabelroot)
    if str_ == 'LGE':
        re['image'] = generate(5,'_LGE.nii.gz',allimageroot)
        re['label'] = generate(5,'_LGE_manual.nii.gz',alllabelroot)
    return re

# load datasets for segmentation
def makedatasets(types:list, lge_valid = True, split=0.2):
    '''
    load image on patient-by-patient basis instead of slices
    types:
        `C0` : bssfp image
        `LGE` : lge image
        `T2`: t2 image
        `C0LGE`: fake lge convert from bssfp
        `T2LGE`: fake lge convert from t2
    Input: 
        `types` -- a list of image type like ['LGE','C0LGE'], which means you want to train or validate
        `lge_valid` -- bool, if ture only `LGE` for validation
                        notice: include `LGE` can take effect, besides the element in `types` must be more than 1(>1) 
        `split` -- float, division ratio of training and test set, besides `split` = valid / all
                        notice: if `lge_valid` is true, this cann't take effect
    Output:
        two numpy of [h,w] for training as image and label input
        and a dict of {'image':[path],'label':[path]} for validation
    '''
    if lge_valid and 'LGE' not in types:
        raise RuntimeError('no LGE type in `types` when `lge_valid` is true')
    if lge_valid and len(types) == 1:
        raise RuntimeError('the element in `types` must more than one when `lge_valid` is true')

    train_image = []
    train_label = []
    valid_path = {}
    paths = {}
    for type_ in types:
        paths[type_] = getfilepath(type_)    
    
    if lge_valid:
        valid_path['image'] = paths['LGE']['image']
        valid_path['label'] = paths['LGE']['label']
        types.remove('LGE')
        for i in types:
            for imagepath in paths[i]['image']:
                out = nii_reader(imagepath)
                train_image.extend(out)
            for labelpath in paths[i]['label']:
                out = nii_reader(labelpath)
                train_label.extend(out)
    else:
        all_imagepath = []
        all_labelpath = []
        for i in types:
            all_imagepath.extend(paths[i]['image'])
            all_labelpath.extend(paths[i]['label'])
        assert len(all_imagepath) == len(all_labelpath)
        lens = int(len(all_imagepath) * split)
        assert lens >= 1
        valid_path['image'] = all_imagepath[:lens]
        valid_path['label'] = all_labelpath[:lens]
        for imagepath in all_imagepath[lens:]:
            out = nii_reader(imagepath)
            train_image.extend(out)
        for labelpath in all_labelpath[lens:]:
            out = nii_reader(labelpath)
            train_label.extend(out)
    return train_image, train_label, valid_path

# Dataloader of Segmentation
class SegDataset(Dataset):
    '''
    Return :
        one dict of {'image':..., 'target':....} for train or validation
        or one dict of {'image':...} for test
    '''
    def __init__(self,transforms_,image, label, mode='train'):
        self.transforms = transforms_
        self.mode = mode
        self.image = image
        self.label = label
        self.mode = mode

    def __getitem__(self,index):
        image = self.image[index]
        mask = self.label[index]
        if self.mode == 'train':
            # do transforms
            data = self.transforms(image=image,mask=mask)
            trans_i = data['image']
            trans_m = data['mask']
            # image normalization
            tensor_i = torch.tensor(trans_i).unsqueeze(dim=0)
            nor_i = min_max(tensor_i)
            # mask convert to onehot
            remark = [[0.0],[200.0],[500.0],[600.0]]
            onehot = mask2onehot(mask=trans_m,label=remark)
            return {'image':nor_i, 'target':onehot}
        elif self.mode == 'valid':
            # resize both ,but only crop image,after test to padding predict
            data = self.transforms(image=image,mask=mask)
            trans_i = data['image']
            trans_m = data['mask']
            tensor_i = torch.tensor(trans_i).unsqueeze(dim=0)
            nor_i = min_max(tensor_i)
            remark = [[0.0],[200.0],[500.0],[600.0]]
            onehot = mask2onehot(mask=trans_m,label=remark)
            return {'image':nor_i, 'target':onehot}

    def __len__(self):
        return(len(self.label))