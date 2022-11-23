# test segmentation model
from EvalAndLoss import DiceScore,JaccardIndex
import torch
import numpy as np
from datasets import *
import SimpleITK as sitk
import argparse
from models import segment_model
from utils import *
from EvalAndLoss import *
import albumentations as A
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

def valid_seg(model=None,dataloader=None,criterion=None):
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
            image = batch['image'].cuda() # [N,1,320,320]
            target = batch['target'].cuda() # [N,4,320,320]
            predict = model(image)
            loss_ = criterion(predict,target).item()
            loss += loss_
            predict = predict.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            # output / label convert to [H,W] 
            output = np.argmax(predict,axis=1).squeeze(axis=0)
            label = np.argmax(target,axis=1).squeeze(axis=0)
            ds = DiceScore(output,label)
            dice.append(ds)
            js = JaccardIndex(output,label)
            jaccard.append(js)
    # compute mean and std
    dice = mean_std(dice)
    jaccard = mean_std(jaccard)
    loss = loss / len(dataloader)
    return {'loss':loss,'dice':dice,'jaccard':jaccard}

##############################################################
# Test all LGE .nii.gz file 
##############################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=4, help='number of channels of output data')
    parser.add_argument('--model_save', type=str, default='output/seg/best_dice.pth',help='path root to store model parameters')
    parser.add_argument('--results', type=str, default='results.txt',help='path to save results')
    parser.add_argument('--model', type=str, default='aunet', help='model to segment')
    parser.add_argument('--histogram_match', type=bool, default=False, help='do histogram match or not')
    parser.add_argument('--size', type=int, default=512, help='size of the data resize')
    parser.add_argument('--centercrop', type=int, default=320, help='size of the data centercrop')
    opt = parser.parse_args()
    print(opt)

    # model
    model = segment_model(opt)
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
    total_overlap =np.zeros((1,4, 5))
    total_surface_distance=np.zeros((1,4, 5))
    # read one paitent and prdict
    for i in range(len(imagename)):
        imageroot = os.path.join('datasets/train/all_image',imagename[i])
        labelroot = os.path.join('datasets/test/C0LET2_gt_for_challenge19/LGE_manual_35_TestData',labelname[i])
        image_ = sitk.ReadImage(imageroot)
        image = sitk.GetArrayFromImage(image_) # [N,H,W]
        out = np.zeros((image.shape[0],image.shape[1],image.shape[2])) # [n,h,w]
        # read one slice and predict
        for j in range(len(image)):
            center = image[j]
            # preprocess
            if opt.histogram_match:
                center = slice_histogram_match(source=center)[0]
            data = A.Compose([
                A.Resize(height=opt.size,width=opt.size),
                A.CenterCrop(height=opt.centercrop,width=opt.centercrop),
            ])(image=center)
            # predict
            input = data['image']
            input = torch.tensor(input).unsqueeze(dim=0) # tensor [1,h,w]
            input = minmax_normal(input).unsqueeze(dim=0).cuda() # [1,1,h,w]
            predict = model(input) # [1,4,h,w]
            output = torch.argmax(predict.detach().cpu(),dim=1).squeeze(dim=0) # [320,320]
            # post-process
            output = reverse_data(output,padding=opt.size,resize=out.shape[1])
            out[j,:,:] = output
        # do validation
        label = sitk.ReadImage(labelroot)
        label = sitk.GetArrayFromImage(label)
        label = (label==200) * 1 + (label==500) * 2 + (label==600) * 3 # [N,h,w]
        dice_score = ThreedDiceScore(out,label)
        result[i,:] = dice_score
        overlap_results,surface_distance_results = Hausdorff_compute(out,label,image_.GetSpacing())
        total_overlap = np.concatenate((total_overlap,overlap_results),axis=0)
        total_surface_distance = np.concatenate((total_surface_distance,surface_distance_results),axis=0)

    np.savetxt(opt.results, result, fmt = '%f', delimiter = ',')
    dicemean = np.mean(result,axis=0)
    dicestd = np.std(result,axis=0)
    mean_overlap = np.mean(total_overlap[1:], axis=0)
    std_overlap = np.std(total_overlap[1:], axis=0)
    mean_surface_distance = np.mean(total_surface_distance[1:], axis=0)
    std_surface_distance = np.std(total_surface_distance[1:], axis=0)
    # write result
    with open (opt.results, 'a') as f:
        f.writelines('Results:\n')
        f.writelines('Dice Score:  Backgroud  ---  Myo  ---  LV  ---  RV\n')
        f.writelines("mean:"+str(dicemean.tolist())+"\n"+"std:"+str(dicestd.tolist())+'\n')
        f.writelines('                  jaccard  ---  dice  ---  volume_similarity  ---  false_negative  ---  false_positive \n')
        f.writelines('Backgroud:mean'+ str(mean_overlap[0].tolist()) + '\n')
        f.writelines('          std'+ str(std_overlap[0].tolist()) + '\n')
        f.writelines('Myo:      mean'+ str(mean_overlap[1].tolist()) + '\n')
        f.writelines('          std'+ str(std_overlap[1].tolist()) + '\n')
        f.writelines('LV:       mean'+ str(mean_overlap[2].tolist()) + '\n')
        f.writelines('          std'+ str(std_overlap[2].tolist()) + '\n')
        f.writelines('RV:       mean'+ str(mean_overlap[3].tolist()) + '\n')
        f.writelines('          std'+ str(std_overlap[3].tolist()) + '\n')
        f.writelines('                  hausdorff_distance  ---  mean_surface_distance  ---  median_surface_distance  ---  std_surface_distance  ---  max_surface_distance:\n')
        f.writelines('Backgroud:mean'+ str(mean_surface_distance[0].tolist()) + '\n')
        f.writelines('          std'+ str(std_surface_distance[0].tolist()) + '\n')
        f.writelines('Myo:      mean'+ str(mean_surface_distance[1].tolist()) + '\n')
        f.writelines('          std'+ str(std_surface_distance[1].tolist()) + '\n')
        f.writelines('LV:       mean'+ str(mean_surface_distance[2].tolist()) + '\n')
        f.writelines('          std'+ str(std_surface_distance[2].tolist()) + '\n')
        f.writelines('RV:       mean'+ str(mean_surface_distance[3].tolist()) + '\n')
        f.writelines('          std'+ str(std_surface_distance[3].tolist()) + '\n')
    print('test done\n')
    print('Dice Score:  Backgroud %.4f ---  Myo %.4f ---  LV %.4f ---  RV %.4f\n'%
        (dicemean[0],dicemean[1],dicemean[2],dicemean[3]))
        
if __name__ == '__main__':
    main()
