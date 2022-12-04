# preprocess class uesed in cyclegan or unet
import torchvision.transforms as transforms
import numpy as np
import albumentations as A
import SimpleITK as sitk
import torch
from typing import Tuple
from scipy.ndimage import distance_transform_edt as eucl_distance
#############################################
# normal transforms
#############################################
class Transformation:
    '''
    ues method .get() to get transformations
    input:
        opt
    output:
        one dict of {'train':..., 'valid':...}
    '''
    def __init__(self, opt):
        self.opt = opt
    def get(self):
        return {
            'cyclegan':self.cyclegan_trans,
            'segmentation':self.segmentation_trans,
        }[self.opt.name]()

    # cyclegan
    def cyclegan_trans(self):
        train_transform = transforms.Compose([
            # just do simple flip
            transforms.RandomHorizontalFlip(),
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(self.opt.size),
        ])
        return {'train':train_transform, 'valid':valid_transform}

    # segmentation
    def segmentation_trans(self):
        train_transforms = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=200,sigma=100,alpha_affine=35,p=0.7),
                A.GridDistortion(p=0.7),
            ],p=0.8),
            A.RandomGamma(gamma_limit=(70,100),p=1),
            A.RandomRotate90(),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Transpose(),
        ])
        valid_transforms = A.Compose([
            # resize and crop. in evaul ,dice score or jaccard index dosen't affect
            # do nothing
        ])
        return {'train':train_transforms, 'valid':valid_transforms}


#########################################################


# rotation based on myo
def Rotation(image,label):
    '''
    Rotation image based on cardic contour
    comes from https://arxiv.org/abs/1910.12514
    Input:
        `image` -- input list of image
        `label` -- matched label
    '''
    lens = len(image)
    for i in range(lens):
        size = image[i].shape[0]
        # get Myo contour
        label = (label==200) * 1 + (label==500) * 0 + (label==600) * 0
        if np.sum(label) == 0:
            # don't have Myo continue
            continue
        
        
# histogram matching
def slice_histogram_match(source:list):
    '''
    histogram match to common one
    implemented by simpleitk tool
    Input:
        `source` -- a list of image [n,h,w]
    Output:
        a numpy [n,h,w]
    '''
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    reference = sitk.ReadImage('datasets/train/fake_lge/patient10_C0_1.nii')
    output = []
    for i in range(len(source)):
        moving = np.expand_dims(source[i],axis=0)
        moving = sitk.GetImageFromArray(moving)
        after = matcher.Execute(moving,reference)
        out = sitk.GetArrayFromImage(after).squeeze(axis=0)
        output.append(out)
    return output

def resample_image(itk_image, is_label=False):
    '''
    Resample image and mask to same resolution ,ji 1.25mm X 1.25mm
    Input:
        `itk_image` -- sitk image
        `is_label` -- is mask
    Output:
        resample sitk image, dtype is 'sitk.image'
    '''
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_spacing = [1.25,1.25,original_spacing[2]]

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
        #original_size[2]
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label: # 如果是mask图像，就选择sitkNearestNeighbor这种插值
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetOutputPixelType(sitk.sitkInt16)
    else: # 如果是普通图像，就采用sitkLiner插值法
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputPixelType(sitk.sitkFloat32)

    return resample.Execute(itk_image) 

# Min_Max normalizatiom
def min_max(input):
    '''
    normalization by slice
    input :
        tensor [1,h,w]
    return :
        minmaxscaler and normalization tensor, range of [-1,1]
    '''
    scaler = (input - torch.min(input))/(torch.max(input) - torch.min(input)) # convert to [0,1]
    normal = (scaler - 0.5) / 0.5 # convert to [-1,1]
    return normal
# z-scores normalization
def zscores(input):
    '''
    z-scores normalization by slice
    input :
        tensor [1,h,w]
    return :
        normal normalization tensor, which means = 0.5, std = 0.5
    '''
    normal = (input - torch.mean(input,dim=(1,2),keepdim=True)) / (torch.std(input,dim=(1,2),keepdim=True))
    out = normal * 0.5 + 0.5
    return out

def truncate(MRI):
    # truncate
    Hist, _ = np.histogram(MRI, bins=int(MRI.max()))

    idexs = np.argwhere(Hist >= 20)
    idex_min = np.float32(0)
    idex_max = np.float32(idexs[-1, 0])

    # MRI[np.where(MRI <= idex_min)] = idex_min
    MRI[np.where(MRI >= idex_max)] = idex_max
    # MRI = MRI - (idex_max+idex_min)/2
    # MRI = MRI / ((idex_max-idex_min)/2)

    # norm
    sig = MRI[0, 0, 0]
    MRI = np.where(MRI != sig, MRI - np.mean(MRI[MRI != sig]), 0 * MRI)
    MRI = np.where(MRI != sig, MRI /
                    np.std(MRI[MRI != sig] + 1e-7), 0 * MRI)
    return MRI

def center_crop(mri):
    '''
    Do center_crop after resample, size of 192X192
    Input:
        `mri` -- resample mri
    '''
    data = A.CenterCrop(height=192,width=192)(image=mri)
    return data['image']

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    '''
    onehot to distance map ,comes from https://proceedings.mlr.press/v102/kervadec19a.html
    Input:
        `seg` -- np, onehot of a label
        `resolution` -- tuple, spacing of real image. This only includes 1.25mm X 1.25mm
    Output:
        distance map of one label
    '''
    #assert one_hot(torch.tensor(seg), axis=0)
    # classes number
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

def np_convert255(mri):
    '''
    min_max normalization and then convert to range of [0,255]
    Input:
        `mri` -- numpy of one slice
    '''
    scale = (mri - np.min(mri)) / (np.max(mri) - np.min(mri))
    normal = scale * 255.0
    return normal

def np_convert1(mri):
    '''
    normalization from [0,255.0] to [-1,1] like torch.transfomer
    '''
    scale = mri / 255.0
    normal = (scale - 0.5) / 0.5
    return normal