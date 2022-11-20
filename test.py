import argparse
import sys
from torch.utils.data import DataLoader
import SimpleITK as sitk
import torch
from datasets import ImageDataset
from models import Generator
from preprocess import Transformation
from utils import denormalization

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/cyclegan/test', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/cyclegan/3/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/cyclegan/3/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--trans_name',type=str, default='cyclegan',help='trans type of dataset')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = Transformation(opt).get()
# got a dict of {bssfp image, label, info}
dataloader = DataLoader(ImageDataset(transforms_=transforms_['valid'], unaligned=True, mode='valid'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

###################################
###### Testing######

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = batch['A'].cuda()
    info = batch['B'][0]
    fake_B = netG_A2B(real_A) # [1,1,256,256]
    # scale to 255
    img = denormalization(fake_B.detach())
    save = sitk.GetImageFromArray(img)
    sitk.WriteImage(save,'datasets/train/t2_lge/%s.nii'%(info))
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
sys.stdout.write('\n')
###################################
