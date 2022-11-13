import random
import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from utils import mask2onehot, minmax_normal

###################################################
# Cyclegan
###################################################
# train dataset of cyclegan
class ImageDataset(Dataset):
    '''
    Dataloader of cyclegan
    '''
    def __init__(self, transforms_ ,unaligned=False):
        self.transform = transforms_
        self.unaligned = unaligned
        self.img_A = nii_loader(str_='T2',is_label=False)
        self.img_B = nii_loader(str_='LGE',is_label=False)
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
        item_A = minmax_normal(item_A)

        if self.unaligned:
            b_tensor = torch.from_numpy(self.img_B[random.randint(0, self.len_b - 1)]).float().unsqueeze(0)
        else:
            b_tensor = torch.from_numpy(self.img_B[index_b]).float().unsqueeze(0)
        item_B = self.transform(b_tensor)
        item_B = minmax_normal(item_B)
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(self.len_a, self.len_b)


# read one .nii file, convert it into numpy 
def nii_reader(datapath):
    '''
    read one .nii.gz or .nii and return all slices
    Input:
        `datapath` -- str_, path of .nii or .nii.gz file
    Output:
        a list of all slice in one patient size is [H, W]
    info:
        original range( > 255.0)
    '''
    re_slice = []
    assert os.path.exists(datapath)
    ori_data = sitk.ReadImage(datapath,outputPixelType=sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(ori_data) # [n,h,w]
    dim = img.shape[0]
    
    for i in range(0,dim):
        img_ = img[i]
        re_slice.append(img_)
    return re_slice

# load all patient images or labels by given str_
def nii_loader(str_, is_label=False):
    '''
    load all patient images or lables by given str_
    str_ includes:
        all : all images or labels
        C0 : all bSSFP images or labels
        T2 : all T2-weight images or labels
        LGE : all LGE CMR or labels
    Input:
        `str_` -- choosed type
        `is_label` -- load label or not
    Output:
        a numpy [N,H,W]
    '''
    dataroot = 'datasets/train/all_image'
    if is_label:
        dataroot = 'datasets/train/all_label'
    filename = os.listdir(dataroot)
    # select filename by str_
    if str_ != 'all':
        filename = list(filter(lambda x: re.search(str_,x) is not None,filename))
    filename.sort(key=lambda x:int(x.split('_')[0][7:]))
    re_images = []
    # get image and info respectively
    for name in filename:
        # get given file all slices
        datapath = os.path.join(dataroot,name)
        images = nii_reader(datapath)
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
    c0lgeroot = 'datasets/train/fake_lge/'
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
    def __init__(self,transforms_,image, label):
        self.transforms = transforms_
        self.image = image
        self.label = label

    def __getitem__(self,index):
        image = self.image[index]
        mask = self.label[index]
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

    def __len__(self):
        return(len(self.label))