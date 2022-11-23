# preprocess class uesed in cyclegan or unet
import torchvision.transforms as transforms
import numpy as np
import albumentations as A
import SimpleITK as sitk
#from datasets import load_image
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
            A.Resize(height=self.opt.size,width=self.opt.size),
            A.CenterCrop(height=self.opt.centercrop,width=self.opt.centercrop),
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
            A.Resize(height=self.opt.size,width=self.opt.size),
            A.CenterCrop(height=self.opt.centercrop,width=self.opt.centercrop)
        ])
        return {'train':train_transforms, 'valid':valid_transforms}


#########################################################
# Image augmentation
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