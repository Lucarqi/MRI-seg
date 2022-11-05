# test segmentation model
from EvalAndLoss import DiceScore,JaccardIndex
import torch
import numpy as np
from datasets import *
import SimpleITK as sitk
import nibabel as nib
from torch.utils.data import DataLoader
from preprocess import Transformation
import argparse
import sys
from models import MUnet
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import *

def mean_std(score):
    '''
    compute mean and std for the given numpy [N,C]
    Input:
        `score` -- scores computed by Dice or Jaccard
    Output:
        a numpy array of [C,2]: dim=0 means classes, dim=1 represent 'mean''std'
    Info:
        Integer `-1` presents on this catagory in both predict and target
        so no need divide this sample
    '''
    score = np.asarray(score)
    re = np.zeros((score.shape[1],2))
    for i in range(0,3):
        item = score[:,i]
        delete = np.delete(item,np.where(item == -1))
        mean = np.mean(delete)
        std = np.std(delete)
        re[[i],:] = np.array([mean,std])
    return re

def valid_seg(model=None,dataloader=None,criterion=None,device=None):
    '''
    model validation
    Input:
        `model` -- model on training or already trained
        `dataloader` -- dataloader of valid/test
        `criterion` -- criterion function
        `device` -- device parameter of cpu or gpu
    Output:
        a dict of {'loss': float ,'dice': [3,2],'jaccard';[3,2]} dim=0 means classes ,dim=1 means 'mean''std'
    '''
    loss = 0
    dice = []
    jaccard = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image = batch['image'].cuda()
            target = batch['target'].cuda()
            predict = model(image)
            loss_ = criterion(predict,target).item()
            loss += loss_
            ds = DiceScore(predict,target)
            dice.append(ds)
            js = JaccardIndex(predict,target)
            jaccard.append(js)
    # compute mean and std
    dice = mean_std(dice)
    jaccard = mean_std(jaccard)
    loss = loss / len(dataloader)
    return {'loss':loss,'dice':dice,'jaccard':jaccard}

##############################################################
# Do test
##############################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=4, help='number of channels of output data')
    parser.add_argument('--save_root', type=str, default='output/seg/best_dice.pth',help='path root to store model parameters')
    parser.add_argument('--trans_name',type=str, default='segmentation',help='trans type of dataset')
    opt = parser.parse_args()
    print(opt)

    # model
    model = MUnet(opt.input_nc,opt.output_nc)
    # load state dict
    model.load_state_dict(torch.load(opt.save_root))
    model.eval()
    # data (ues )
    types = ['LGE']
    data = load_image(str=types,paired_label=True)
    index = 10
    image = data['image'][index]
    label = data['label'][index]
    input = torch.tensor(image).unsqueeze(dim=0)
    nor_i = minmax_normal(input).unsqueeze(dim=0)
    # do test
    predict = model(nor_i)
    output = F.softmax(predict.detach(),dim=1).squeeze(dim=0)
    remark = [[0.0],[200.0],[500.0],[600.0]]
    label_ = onehot2mask(output,remark)
    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.imsave(image,'output/seg/image.png')
    plt.subplot(1,3,2)
    plt.imsave(label,'output/seg/label.png')
    plt.subplot(1,3,3)
    plt.imsave(label_,'output/seg/seg.png')
if __name__ == '__main__':
    main()
