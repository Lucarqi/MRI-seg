# loss and evaluation function 
import torch.nn as nn
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np

###################################################
# Loss function
###################################################

# convert onehot mask to index mask
def onehot2index(onehot):
    '''
    Input:
        `onehot` -- a tensor onehot mask like [N, C, H, W]
    Output:
        `index` -- a tensor index mask like [N, 1, H, W]
    '''
    index = torch.argmax(onehot,dim=1,keepdim=True)
    return index

class CrossEntropyLoss(nn.Module):
    '''
    Loss function like torch.nn.CrossEntropy
    Input:
        `predict` -- a tensor of [N, C, H, W]
        `target` -- a tensor onehot mask that shape as `predict`
    '''
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss,self).__init__()
        self.reduction = reduction
    def forward(self,predict,target):
        # convert onehot to index mask
        index = onehot2index(target)
        index = torch.squeeze(index,dim=1)
        # apply log_softmax = softmax then log [N,C,H,W]
        log = F.log_softmax(predict,dim=1)
        # apply nll_loss , loss is [1, ] scaler
        loss = F.nll_loss(log,index,reduction=self.reduction)
        return loss

class FocalLoss(nn.Module):
    '''
    FocalLoss by 
    Input:
        `predict` -- a tensor of [N, C, H, W]
        `target` -- a tensor onehot mask that shape as `predict`
    '''
    def __init__(self,gamma=0,eps=1e-7,reduction='mean'):
        super(FocalLoss,self).__init__()
        self.eps = eps
        self.gamma = gamma
        self.reduction = reduction
        self.ce = CrossEntropyLoss(reduction='none')
    def forward(self,predict,target):
        # get -logp
        nlogp = self.ce(predict,target)
        # get p
        p = torch.exp(-nlogp).clamp(min=0.00001,max=0.99999)
        # get focal loss = -(1-p)^gamma * logp
        loss = (1+self.eps-p)**self.gamma * nlogp
        if self.reduction == 'mean':
            return torch.mean(loss)
        return loss

class BinaryDiceLoss(nn.Module):
    '''
    Soft Binary Dice Loss by
    Input:
        `smooth` -- a float number to smooth loss, avoid Nan error, default:1.0
        `p` -- Denominator value: \sum{x^p} + \sum{y^p}, default:1
        `reduction` -- 'none'|'mean' for reduction, default:mean
        `predict` -- a tensor of [N, 1, H, W]
        `target` -- a tensor onehot mask that shape as `predict`
    Info:
        no softmax or sigmoid in this function
    '''
    def __init__(self, smooth=1, p=1, reduction='mean'):
        super(BinaryDiceLoss,self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
    def forward(self,predict, target):
        # flatten to [N,H*W]
        predict = predict.contiguous().view(predict.shape[0],-1)
        target = target.contiguous().view(predict.shape[0],-1).float()
        # |X jiao Y|
        x_y = torch.sum(predict * target, dim=1)
        # |X|
        x = torch.sum(predict,dim=1)
        # |Y|
        y = torch.sum(target,dim=1)
        # loss
        dice = (2*x_y + self.smooth) / (x.pow(self.p) + y.pow(self.p) + self.smooth)
        if self.reduction == 'mean':
            return 1 - torch.mean(dice) 
        return 1 - dice

class DiceLoss(nn.Module):
    ''' 
    DiceLoss (Multi Binary DiceLoss) 
    Input:
        `kwargs` -- args used in BinaryDiceLoss
        `predict` -- a tensor of [N, C, H, W]
        `target` -- a tensor onehot mask that shape as `predict`
    '''
    def __init__(self, **kwargs):
        super(DiceLoss,self).__init__()
        self.kwargs = kwargs

    def forward(self,predict,target):
        assert predict.shape == target.shape
        nclasses = predict.shape[1]
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict,dim=1)
        # compute DiceLoss = compute BinaryDiceLoss one by one layer separately
        for i in range(0,nclasses):
            dice_loss = dice(predict[:,i],target[:,i])
            total_loss = total_loss + dice_loss
        return total_loss / nclasses

##################################################
# evaluation function
##################################################

def DiceScore(predict, target, nclasses=4):
    '''
    2D Dice Score for validation step
    Input:
        `predict` -- a numpy of prediction [H,W]
        `target` -- a numpy of target onehot [H,W]
        `nclasses` -- classes need to segmentate
    Output:
        a list as Dice Score of this sample in order of {'Myo','LV','RV'}
    Info:
        there are four class as {Backgroud,Myo,LV,RV}
        value '-1' means this class doesn't exesits in both `predict` and `target`
    '''
    remarks = [0,1,2,3]
    bk_remark = 0
    dice = []
    for i in range(nclasses):
        # no need compute 'Backgroud'
        if remarks[i] == bk_remark:
            continue
        intersection = np.sum((predict==remarks[i]) * (target==remarks[i]))
        union = np.sum(predict==remarks[i]) + np.sum(target==remarks[i])
        if union==0:
            # means there is no this class, return `-1`
            dice.append(-1)
        else:
            dice.append(2*intersection / union)
    assert len(dice) == nclasses-1
    return dice
    
def JaccardIndex(predict, target, nclasses=4):
    '''
    2D Jaccard Index for validation step
    Input:
        `predict` -- a numpy of prediction [H,W]
        `target` -- a numpy of target onehot [H,W]
        `nclasses` -- classes need to segmentate
    Output:
        a list as JaccardIndex of this sample in order of {'Myo','LV','RV'}
    Info:
        there are four class as {Backgroud,Myo,LV,RV}
        value '-1' means this class doesn't exesits in both `predict` and `target`
    '''
    remarks = [0,1,2,3]
    bk_remark = 0
    jaccard = []
    for i in range(nclasses):
        # no need compute 'Backgroud'
        if remarks[i] == bk_remark:
            continue
        intersection = np.sum((predict==remarks[i]) * (target==remarks[i]))
        union = np.sum(predict==remarks[i]) + np.sum(target==remarks[i]) - intersection
        if union==0:
            # means there is no this class, return `-1`
            jaccard.append(-1)
        else:
            jaccard.append(intersection / union)
    assert len(jaccard) == nclasses-1
    return jaccard

def ThreedDiceScore(predict,target):
    '''
    3D Dice Score
    Input:
        `predict` -- a numpy of prediction [batchsize,H,W]
        `target` -- a numpy of target mask [batchsize,H,W]
    Output:
        a numpy [4,] as DiceScore of this sample in order of {'background','Myo','LV','RV'} 
    '''
    dice = []
    for i in range(0,4):
        dice_i = 2*(np.sum((predict==i)*(target==i),dtype=np.float32)+0.0001)/(np.sum(predict==i,dtype=np.float32)+np.sum(target==i,dtype=np.float32)+0.0001)
        dice = dice + [dice_i]
    return np.array(dice,dtype=np.float32)

def Hausdorff_compute(pred,groundtruth,spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1,4, 5))
    surface_distance_results = np.zeros((1,4, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(4):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)


    return overlap_results,surface_distance_results
