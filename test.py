import argparse
import sys
import re
import os
import torch
import numpy as np
import albumentations as A
from models import Generator
from preprocess import min_max
from utils import saveasnii
from datasets import nii_reader
parser = argparse.ArgumentParser()
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/cyclegan/4/netG_A2B.pth', help='A2B generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))

# Set model's test mode
netG_A2B.eval()

# Get data file name
allname = os.listdir('datasets/train/all_image')
name = list(filter(lambda x: re.search('C0',x) is not None,allname))

###################################
###### Testing######
for i in range(len(name)):
    dataroot = os.path.join('datasets/train/all_image',name[i])
    assert os.path.exists(dataroot)
    image = nii_reader(dataroot)
    out = np.zeros((len(image),image[1].shape[0],image[1].shape[1]))
    # generate fake lge slice by slice
    for j in range(len(image)):
        # resize 
        idata = A.Resize(height=256,width=256)(image=image[j])
        input = idata['image']
        input = torch.tensor(input).unsqueeze(dim=0)
        input = min_max(input).unsqueeze(dim=0).cuda()
        fake_lge = netG_A2B(input)
        output = fake_lge.detach().cpu().numpy().squeeze() # [h,w]
        # reverse resize
        odata = A.Resize(height=out.shape[1],width=out.shape[2])(image=output)
        output = odata['image']
        out[j] = output
    # save as .nii.gz and range of [0,255]
    saveasnii(image=out,info=name[i])
    sys.stdout.write('\rGenerated patient image %s'%(name[i]))
sys.stdout.write('\n')
###################################
