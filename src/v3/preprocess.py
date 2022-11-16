# preprocess class uesed in cyclegan or unet
import numpy as np
import albumentations as A
import SimpleITK as sitk
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
        train_transform = A.Compose([
            A.Resize(int(self.opt.size*1.2),int(self.opt.size*1.2), cv2.INTER_LINEAR), 
            A.RandomCrop(self.opt.size,self.opt.size), 
            A.HorizontalFlip(),
        ])
        valid_transform = A.Compose([
            A.Resize(self.opt.size,self.opt.size),
        ])
        return {'train':train_transform, 'valid':valid_transform}

    # segmentation
    def segmentation_trans(self):
        train_transforms = A.Compose([
            A.Resize(height=512,width=512),
            A.CenterCrop(height=320,width=320),
            #A.ElasticTransform(alpha=200,sigma=100,alpha_affine=35,p=0.5),
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
class MyoRotate():
    def __init__(self) -> None:
        pass

# histogram matching
def slice_histogram_match(source:list):
    '''
    histogram match to common one
    implemented by simpleitk tool
    Input:
        `source` -- a list of image [n,h,w]
    Output:
        a list size of [h,w]
    '''
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    reference = sitk.ReadImage('datasets/train/match/match.nii.gz')
    output = []
    for i in range(len(source)):
        moving = np.expand_dims(source[i],axis=0)
        moving = sitk.GetImageFromArray(moving)
        after = matcher.Execute(moving,reference)
        out = sitk.GetArrayFromImage(after).squeeze(axis=0)
        output.append(out)
    return output

def cyc_histogram_match(source:list):
    '''
    histogram match used in cyclegan
    Input:
        `source` -- source image list
    Output:
        a numpy of [h,w]
    '''
    # do resize
    trans =  A.Resize(512,512)
    input = []
    data = trans(image=source[0])
    input.append(data['image'])
    return slice_histogram_match(input)