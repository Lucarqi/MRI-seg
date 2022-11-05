# loss and evaluation function 
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
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

def DiceScore(predict, target):
    '''
    Dice Score for validation step
    Input:
        `predict` -- a tensor of prediction [1,C,H,W]
        `target` -- a tensor of target onehot [1,C,H,W]
    Output:
        a list as Dice Score of this sample in order of {'LV','Myo','RV'}
    Info:
        there are four class as {Backgroud,LV,Myo,RV}
        value '-1' means this class doesn't exesits in both `predict` and `target`
    '''
    nclasses = predict.shape[1]
    remarks = [0,1,2,3]
    bk_remark = 0
    if torch.is_tensor(predict):
        output = F.softmax(predict,dim=1).detach().cpu().numpy()
    if torch.is_tensor(target):
        label = target.detach().cpu().numpy()
    # output / label convert to [H,W] 
    output = np.argmax(output,axis=1).squeeze(axis=0)
    label = np.argmax(label,axis=1).squeeze(axis=0)
    dice = []
    for i in range(nclasses):
        # no need compute 'Backgroud'
        if remarks[i] == bk_remark:
            continue
        intersection = np.sum((output==remarks[i]) * (label==remarks[i]))
        union = np.sum(output==remarks[i]) + np.sum(label==remarks[i])
        if union==0:
            # means there is no this class, return `-1`
            dice.append(-1)
        else:
            dice.append(2*intersection / union)
    assert len(dice) == nclasses-1
    return dice
    
def JaccardIndex(predict, target):
    '''
    Jaccard Index for validation step
    Input:
        `predict` -- a tensor of prediction [1,C,H,W]
        `target` -- a tensor of target onehot [1,C,H,W]
    Output:
        a list as JaccardIndex of this sample in order of {'LV','Myo','RV'}
    Info:
        there are four class as {Backgroud,LV,Myo,RV}
        value '-1' means this class doesn't exesits in both `predict` and `target`
    '''
    nclasses = predict.shape[1]
    remarks = [0,1,2,3]
    bk_remark = 0
    if torch.is_tensor(predict):
        output = F.softmax(predict,dim=1).detach().cpu().numpy()
    if torch.is_tensor(target):
        label = target.detach().cpu().numpy()
    # output / label convert to [H,W] 
    output = np.argmax(output,axis=1).squeeze(axis=0)
    label = np.argmax(label,axis=1).squeeze(axis=0)
    jaccard = []
    for i in range(nclasses):
        # no need compute 'Backgroud'
        if remarks[i] == bk_remark:
            continue
        intersection = np.sum((output==remarks[i]) * (label==remarks[i]))
        union = np.sum(output==remarks[i]) + np.sum(label==remarks[i]) - intersection
        if union==0:
            # means there is no this class, return `-1`
            jaccard.append(-1)
        else:
            jaccard.append(intersection / union)
    assert len(jaccard) == nclasses-1
    return jaccard
    