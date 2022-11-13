# preprocess class uesed in cyclegan or unet
import torchvision.transforms as transforms
import numpy as np
import torch
import albumentations as A
import cv2

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
        self.name = opt.trans_name
        self.opt = opt
        
    def get(self):
        return {
            'cyclegan':self.cyclegan_trans,
            'segmentation':self.segmentation_trans,
        }[self.name]()

    # cyclegan
    def cyclegan_trans(self):
        train_transform = transforms.Compose([
            transforms.Resize(int(self.opt.size*1.2), transforms.InterpolationMode.BILINEAR), 
            transforms.RandomCrop(self.opt.size), 
            transforms.RandomHorizontalFlip(),
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(self.opt.size),
        ])
        return {'train':train_transform, 'valid':valid_transform}

    # segmentation
    def segmentation_trans(self):
        train_transforms = A.Compose([
            A.Resize(height=512,width=512),
            A.CenterCrop(height=320,width=320),
            A.ElasticTransform(alpha=200,sigma=100,alpha_affine=35,p=0.7),
            A.RandomRotate90(),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.Transpose(),
        ])
        valid_transforms = A.Compose([
            # resize and crop. in evaul ,dice score or jaccard index dosen't affect
            A.Resize(height=512,width=512),
            A.CenterCrop(height=320,width=320)
        ])
        return {'train':train_transforms, 'valid':valid_transforms}


#########################################################
# Image augmentation
#########################################################

# rotation based on myo
class MyoRotate:
    def __init__(self) -> None:
        pass
