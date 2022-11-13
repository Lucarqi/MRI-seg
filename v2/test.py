import argparse
import sys
import numpy as np
import os
import re
import torch
import SimpleITK as sitk
from utils import saveasnii, minmax_normal
from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='datasets/train/all_image', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--generator_A2B', type=str, default='output/cyclegan/3/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--source_type', type=str, default='T2', help='source image type ,include C0 or T2')
opt = parser.parse_args()
print(opt)

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_A2B.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))

# Set model's test mode
netG_A2B.eval()

# Get data file name
allname = os.listdir(opt.dataroot)
name = list(filter(lambda x: re.search(opt.source_type,x) is not None,allname))

###################################
###### Testing######
for i in range(len(name)):
    dataroot = os.path.join(opt.dataroot,name[i])
    source = sitk.ReadImage(dataroot,outputPixelType=sitk.sitkFloat32)# [h,w,n]
    image = sitk.GetArrayFromImage(source) # [n,h,w]
    out = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    # generate fake lge slice by slice
    for j in range(image.shape[0]):
        input = torch.tensor(image[j]).unsqueeze(dim=0)
        input = minmax_normal(input).unsqueeze(dim=0).cuda()
        fake_lge = netG_A2B(input)
        output = fake_lge.detach().cpu().numpy().squeeze() # [h,w]
        out[j,:,:] = output
    # save as .nii.gz
    saveasnii(image=out,info=name[i])
    sys.stdout.write('\rGenerated patient image %s'%(name[i]))
sys.stdout.write('\n')
###################################
