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

def valid_seg(model=None,datapath=None,type=None,a=None):
    '''
    model validation
    Input:
        `model` -- model on training or already trained
        `datapath` -- dict of {'image':[path],'label':[path]}, validation datasets file path
        `type` -- criterion function type
        `a` -- parameter of boundary loss
    Output:
        a dict of {'loss': float ,'dice': [3,2]} dim=0 means classes ,dim=1 means 'mean''std'
    '''
    model.eval()
    with torch.no_grad():
        if type == 'bdloss':
            criterion, boundary_loss = CrossEntropyLoss(),BoundaryLoss(idc=[1,2,3])
        else:
            criterion = init_criterion(init_type=criterion)
        num = len(datapath['image'])
        loss = 0.0
        lens = 0
        result = []
        for i in range(num):
            imagepath = datapath['image'][i]
            labelpath = datapath['label'][i]
            image = load_patient(imagepath,is_label=False)
            label = load_patient(labelpath,is_label=True)
            imageout = np.zeros((len(image),192,192)) # [n,h,w]
            labelout = np.zeros((len(label),192,192))
            lens = lens + len(image)
            # get all predict
            for j in range(len(image)):
                input = image[j]
                input = np_convert1(input)
                mask = label[j]
                input = torch.tensor(input,dtype=torch.float32).unsqueeze(dim=0)
                input = input.unsqueeze(dim=0).cuda() # [1,1,192,192]
                remark = [[0.0],[200.0],[500.0],[600.0]]
                target = mask2onehot(mask=mask,label=remark).unsqueeze(dim=0).cuda() # [1,4,192,192]
                seg = target.numpy()
                dist_map = one_hot2dist(seg,resolution=[1.25,1.25],dtype=np.float32)
                dist_map = torch.tensor(dist_map)
                predict = model(input) # [1,4,192,192]
                loss_ = criterion(predict, target)
                if type == 'bdloss':
                    loss_ = loss_ + a * boundary_loss(predict,dist_map)
                loss = loss + loss_.item()
                output = torch.argmax(predict.detach().cpu(),dim=1).squeeze(dim=0) # [192,192]
                imageout[j,:,:] = output
                labelout[j,:,:] = (mask==200) * 1 + (mask==500) * 2 + (mask==600) * 3
            # do dice score
            score = ThreedDiceScore(imageout,labelout)
            result.append(score)
        # return 
        avg_loss = loss / lens
        scores = np.mean(np.array(result),axis=0).flatten()
        return avg_loss, scores

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
    total_overlap =np.zeros((1,4,5))
    total_surface_distance=np.zeros((1,4, 5))
    # read one paitent and prdict
    for i in range(len(imagename)):
        imageroot = os.path.join('datasets/train/all_image',imagename[i])
        labelroot = os.path.join('datasets/test/C0LET2_gt_for_challenge19/LGE_manual_35_TestData',labelname[i])
        image_ = sitk.ReadImage(imageroot)
        ori_shape = image_.GetSize()[1]
        # do resample and centercrop
        image_ = resample_image(image_, is_label=False)
        image = sitk.GetArrayFromImage(image_) 
        out = np.zeros((image.shape[0],image.shape[1],image.shape[2])) # [N,192,192]
        # read one slice and predict
        for j in range(len(image)):
            # predict
            input = image[j]
            input = torch.tensor(input).unsqueeze(dim=0) 
            input = min_max(input).unsqueeze(dim=0).cuda() 
            predict = model(input) 
            output = torch.argmax(predict.detach().cpu(),dim=1).squeeze(dim=0) 
            # post-process
            output = reverse_data(output,padding=out.shape[1],resize=ori_shape)
            out[j,:,:] = output
        # do validation
        label = sitk.ReadImage(labelroot)
        label = sitk.GetArrayFromImage(label)
        label = (label==200) * 1 + (label==500) * 2 + (label==600) * 3 
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
