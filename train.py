import argparse
import itertools

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import os

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import init_weights
from datasets import ImageDataset
from preprocess import *
from utils import set_requires_grad

# 超参数的设置
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/cyclegan/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=192, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--lambda_A', type=float, default=10.0, help='intensity of D_A loss')
parser.add_argument('--lambda_B', type=float, default=10.0, help='intensity of D_B loss')
parser.add_argument('--lambda_idt', type=float, default=0.5, help='intensity of identity loss')
parser.add_argument('--save_root', type=str, default='output/cyclegan/7', help='loss path to save')
parser.add_argument('--name', type=str, default='cyclegan', help='chooes transformation type (cyclegan or segmentation)')
parser.add_argument('--source_domain',type=str, default='C0',help='source domain')
opt = parser.parse_args()
print(opt)

if os.path.exists(opt.save_root):
    pass
else:
    os.mkdir(opt.save_root)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

init_weights(net=netG_A2B,init_type='normal')
init_weights(net=netG_B2A,init_type='normal')
init_weights(net=netD_A,init_type='normal')
init_weights(net=netD_B,init_type='normal')

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(netD_A.parameters(), netD_B.parameters()), 
                                lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# Target's size is Discriminator net out (192 -> 22) that full with 1.0 for True or 0.0 for False
target_real = Variable(Tensor(1,1,22,22).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(1,1,22,22).fill_(0.0), requires_grad=False)
# Fake image pool
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
# Data Transforms
transforms_ = Transformation(opt).get()

# Dataset loader
dataloader = DataLoader(ImageDataset(transforms_['train'], opt), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
# Loss plot realtime
loss_save = os.path.join(opt.save_root,'loss.csv')
logger = Logger(opt.n_epochs, len(dataloader), loss_save)

###################################
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Model Forward
        fake_B = netG_A2B(real_A) # G_A2B(A)
        rec_A = netG_B2A(fake_B)  # G_B2A(G_A2B(A))
        fake_A = netG_B2A(real_B) # G_B2A(B)
        rec_B = netG_A2B(fake_A)  # G_A2B(G_B2A(B))

        ###### Generators A2B and B2A ######
        # lock D_A & D_B gradints
        set_requires_grad([netD_A,netD_B],False)

        optimizer_G.zero_grad()
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*opt.lambda_B*opt.lambda_idt
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*opt.lambda_A*opt.lambda_idt

        # GAN loss
        pred_fake_B = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)

        pred_fake_A = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)

        # Cycle loss
        loss_cycle_ABA = criterion_cycle(rec_A, real_A)*opt.lambda_A
        loss_cycle_BAB = criterion_cycle(rec_B, real_B)*opt.lambda_B

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator B ######
        # unlock D_A & D_B gradints
        set_requires_grad([netD_A,netD_B],True)
        optimizer_D.zero_grad()

        # Real B loss
        pred_real_B = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real_B, target_real)

        # Fake B loss (randomly select fake A image)
        fake_B_pool = fake_B_buffer.push_and_pop(fake_B)
        fake_B_pool = netD_B(fake_B_pool.detach())
        loss_D_fake = criterion_GAN(fake_B_pool, target_fake)

        # Total B loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()


        ###### Discriminator A ######

        # Real A loss
        pred_real_A = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real_A, target_real)
        
        # Fake A loss (randomly select fake B image)
        fake_A_pool = fake_A_buffer.push_and_pop(fake_A)
        fake_A_pool = netD_B(fake_A_pool.detach())
        loss_D_fake = criterion_GAN(fake_A_pool, target_fake)

        # Total A loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D.step()
        ###################################

        # Progress report
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), os.path.join(opt.save_root,'netG_A2B.pth'))
    torch.save(netG_B2A.state_dict(), os.path.join(opt.save_root,'netG_B2A.pth'))
    torch.save(netD_A.state_dict(), os.path.join(opt.save_root,'netD_A.pth'))
    torch.save(netD_B.state_dict(), os.path.join(opt.save_root,'netD_B.pth'))
###################################
