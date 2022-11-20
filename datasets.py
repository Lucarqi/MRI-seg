import random
import os
import re

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import SimpleITK as sitk
import albumentations as A
from utils import mask2onehot, minmax_normal
from preprocess import slice_histogram_match

###################################################
# Cyclegan
###################################################
# train dataset of cyclegan
class ImageDataset(Dataset):
    '''
    Dataloader of cyclegan
    '''
    def __init__(self, transforms_ ,unaligned=False, mode='train'):
        self.transform = transforms_
        self.unaligned = unaligned
        data = nii_loader(str='T2',is_label=False)
        self.img_A = data['image']
        self.img_B = nii_loader(str='LGE',is_label=False)['image']
        self.info_A = data['info']
        self.len_a = len(self.img_A)
        self.len_b = len(self.img_B)
        self.mode = mode

    def __getitem__(self, index):
        # warning: make sure index does't out of range
        index_a = index % self.len_a
        index_b = index % self.len_b 
        # apply torch transform
        a_tensor = torch.from_numpy(self.img_A[index_a]).float().unsqueeze(0)
        item_A = self.transform(a_tensor)
        # minmaxscaler and normalization
        item_A = minmax_normal(item_A)

        if self.unaligned:
            b_tensor = torch.from_numpy(self.img_B[random.randint(0, self.len_b - 1)]).float().unsqueeze(0)
        else:
            b_tensor = torch.from_numpy(self.img_B[index_b]).float().unsqueeze(0)
        item_B = self.transform(b_tensor)
        item_B = minmax_normal(item_B)
        # B is info if mode='valid'
        if self.mode == 'valid':
            info_ = self.info_A[index_a]
            return {'A':item_A,'B':info_}
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        if self.mode =='valid':
            return self.len_a
        return max(self.len_a, self.len_b)


# read .nii file, convert it into numpy 
def nii_reader(dataroot,filename):
    '''
    Method: 
        read one .nii.gz or .nii and return all slices
    
    Parameters:
        dataroot: 
            path of .nii or .nii.gz file
        filename:

    Return:
        re_slice: 
            all slices of given file , dtype is numpy.float32 list, size is [N, H, W]
            original range( > 255.0)
        info:
            basic info of given file , dtype is string list, size is [N, ]
    info:
        there is no none pixel image in LGE, C0
    '''

    re_slice = []
    info = []
    root = os.path.join(dataroot,filename)
    assert os.path.exists(root)
    ori_data = nib.load(root)
    # convert float64 into float32 ,float64 does not supported by transform.ToPILiamge
    img = np.float32(ori_data.get_fdata())
    dim = img.shape[-1]
    
    for i in range(0,dim):
        img_ = img[:,:,i]
        info_ = re.search('([\w]*).nii([.]*)',filename).group(1) + '_' + str(i+1)
        re_slice.append(img_)
        info.append(info_)

    return re_slice, info

# load all patient images or labels by given str
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
        str:
            choosed type
        is_label:
            load label or not
    Return :
        one dict of {'image':[N, H, W],'info':[N,]}
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
    re_infos = []
    # get image and info respectively
    for name in filename:
        # get given file all slices
        images, infos = nii_reader(dataroot,name)
        re_images.extend(images)
        re_infos.extend(infos)
    
    return {'image':re_images, 'info':re_infos}

###############################################
# Segmentation
###############################################

# 加载fake_lge数据
def load_fake_lge(type=None):
    '''
    load all fake lge that convert from bssfp
    input:
        `type` -- source image
    output:
        a list of all fake lge [N, H, W]
    '''
    re_image = []
    root = {'C0LGE':'datasets/train/fake_lge','T2LGE':'datasets/train/t2_lge'}
    root = root[type]
    filename = os.listdir(root)
    filename.sort(key=lambda x:(int(re.split(r'_|\.',x)[0][7:]), int(re.split(r'_|\.',x)[2])))
    for name in filename:
        data_dir = os.path.join(root,name)
        d1 = sitk.ReadImage(data_dir)
        d2 = sitk.GetArrayFromImage(d1)
        # original size
        fake_lge = np.squeeze(d2)
        re_image.append(fake_lge)
    return re_image

# 为分割任务加载数据
def load_image(str, paired_label=True, histogram_match=False):
    '''
    load paired {image,label} or {image, (no label)} based on 'str' and 'paired_label' for segmentation
    str is the type of image , include :
        `LGE` -- LGE MRI
        `T2` -- T2 MRI
        `C0` -- bSSFP MRI
        `C0LGE` -- fake lge convert from bssfp
        `T2LGE` -- fake lge convert from t2
    input:
        `str` -- a list of need type, like ['LGE','C0LGE']
        `paired_label` -- a bool, if True only return image that exesits label,otherwise only return image doesn't have label
        `histogram_match` -- a bool, whether match MRI to common one
                            notice: 'common one' is choosed by author , which is easly naked-eye-distinguishable
                                    and this option can be only used in LGE or fake-LGE modality
    output:
        one dict of {'image':..., 'label':...}
        or one dict of {'image':...}
    '''
    label = []
    image = []
    for type_ in str:
        data = []
        label_ = []
        image_ = []
        if type_ == 'C0LGE' or type_ == 'T2LGE':
            data = load_fake_lge(type_)
            str = {'C0LGE':'C0','T2LGE':'T2'}
            label_ = nii_loader(str=str[type_],is_label=True)['image']
        else:
            data = nii_loader(str=type_,)['image']
            label_ = nii_loader(str=type_,is_label=True)['image']
        if paired_label:
            # pick images which have label
            image_ = data[:len(label_)]
        else:
            # pick images which don't have label
            image_ = data[len(label_):]
        image.extend(image_)
        label.extend(label_)
    if histogram_match:
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(7)
        matcher.ThresholdAtMeanIntensityOn()
        reference = sitk.ReadImage('datasets/train/fake_lge/patient10_C0_1.nii')
        image = slice_histogram_match(source=image,reference=reference,filter=matcher)
        
    if paired_label:
        return {'image':image, 'label':label}
    else:
        return {'image':image}

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
            nor_i = minmax_normal(tensor_i)
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
            nor_i = minmax_normal(tensor_i)
            remark = [[0.0],[200.0],[500.0],[600.0]]
            onehot = mask2onehot(mask=trans_m,label=remark)
            return {'image':nor_i, 'target':onehot}

    def __len__(self):
        return(len(self.label))