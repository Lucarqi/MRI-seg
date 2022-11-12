# test segmentation model
import torch
import numpy as np
from datasets import *
import SimpleITK as sitk
import argparse
from models import Unet
from utils import *
from EvalAndLoss import *
import albumentations as A

def valid_seg(model=None,datapath=None,criterion=None):
    '''
    model validation
    Input:
        `model` -- model on training or already trained
        `datapath` -- dict of {'image':[path],'label':[path]}, validation datasets file path
        `criterion` -- criterion function
    Output:
        a dict of {'loss': float ,'dice': [3,2]} dim=0 means classes ,dim=1 means 'mean''std'
    '''
    model.eval()
    with torch.no_grad():
        num = len(datapath['image'])
        loss = 0.0
        lens = 0
        result = []
        for i in range(num):
            imagepath = datapath['image'][i]
            labelpath = datapath['label'][i]
            image = sitk.ReadImage(imagepath)
            image = sitk.GetArrayFromImage(image) # [N,H,W]
            label = sitk.ReadImage(labelpath)
            label = sitk.GetArrayFromImage(label) # [N,H,W]
            imageout = np.zeros((image.shape[0],320,320)) # [n,h,w]
            labelout = np.zeros((image.shape[0],320,320))
            lens = lens + len(image)
            # get all predict
            for j in range(len(image)):
                # do center crop
                data = A.Compose(A.Resize(height=512,width=512),
                                A.CenterCrop(height=320,width=320))(image=image[j],mask=label[j])
                input = data['image']
                mask = data['mask']
                input = torch.tensor(input).unsqueeze(dim=0)
                input = minmax_normal(input).unsqueeze(dim=0).cuda() # [1,1,320,320]
                remark = [[0.0],[200.0],[500.0],[600.0]]
                target = mask2onehot(mask=data['mask'],label=remark).unsqueeze(dim=0).cuda() # [1,4,320,320]
                predict = model(input) # [1,4,320,320]
                loss_ = criterion(loss_, target).item()
                loss = loss + loss_
                output = torch.argmax(predict.detach().cpu(),dim=1).squeeze(dim=0) # [320,320]
                imageout[j,:,:] = output
                labelout[j,:,:] = (mask==200) * 1 + (mask==500) * 2 + (mask==600) * 3
            # do dice score
            score = ThreedDiceScore(imageout,labelout)
            result.append(score)
        # return 
        avg_loss = loss / lens
        scores = np.mean(np.array(score),axis=0).flatten()
        return avg_loss, scores

##############################################################
# Test all LGE .nii.gz file 
##############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=4, help='number of channels of output data')
    parser.add_argument('--model_save', type=str, default='output/seg/best_dice.pth',help='path root to store model parameters')
    opt = parser.parse_args()
    print(opt)

    # model
    model = Unet(opt.input_nc,opt.output_nc)
    # load state dict
    model.load_state_dict(torch.load(opt.model_save))
    model.eval().cuda()
    # load 6-45 LGE patient 
    imagename = os.listdir('datasets/train/all_image')
    imagename = list(filter(lambda x: re.search('LGE',x) is not None,imagename)) # choose LGE
    imagename = list(filter(lambda x: int(x.split('_')[0][7:]) > 5, imagename)) # choose > 5
    imagename.sort(key=lambda x:int(x.split('_')[0][7:])) # sort
    # load 6-45 label
    labelname = os.listdir('datasets/test/C0LET2_gt_for_challenge19/LGE_manual_35_TestData')
    labelname.sort(key=lambda x:int(x.split('_')[0][7:]))
    result = np.zeros((len(imagename),4))
    for i in range(len(imagename)):
        imageroot = os.path.join('datasets/train/all_image',imagename[i])
        labelroot = os.path.join('datasets/test/C0LET2_gt_for_challenge19/LGE_manual_35_TestData',labelname[i])
        image = sitk.ReadImage(imageroot)
        image = sitk.GetArrayFromImage(image) # [N,H,W]
        out = np.zeros((image.shape[0],image.shape[1],image.shape[2])) # [n,h,w]
        # get all predict
        for j in range(len(image)):
            # do center crop
            data = A.CenterCrop(height=320,width=320)(image=image[j])
            input = data['image']
            input = torch.tensor(input).unsqueeze(dim=0) # tensor [1,h,w]
            input = minmax_normal(input).unsqueeze(dim=0).cuda() # [1,1,h,w]
            predict = model(input) # [1,4,h,w]
            output = torch.argmax(predict.detach().cpu(),dim=1).squeeze(dim=0) # [320,320]
            # padding zeros
            output = reverse_centercrop(output,size=out.shape[1])
            out[j,:,:] = output

        label = sitk.ReadImage(labelroot)
        label = sitk.GetArrayFromImage(label)
        label = (label==200) * 1 + (label==500) * 2 + (label==600) * 3 # [N,h,w]
        dice_score = ThreedDiceScore(out,label)
        result[i,:] = dice_score
    np.savetxt("result.txt", result, fmt = '%f', delimiter = ',')
    mean = np.mean(result,axis=0)
    print('Dice Score:  Myo:%.4f  --  LV:%.4f  --  RV:%.4f'%(mean[1],mean[2],mean[3]))
