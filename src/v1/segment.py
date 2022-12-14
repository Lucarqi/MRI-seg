# segmentation train step
import argparse
from torch.utils.data import DataLoader
import torch
import os

from models import MUnet,Unet
from torch.autograd import Variable
from utils import LambdaLR, Seglogger
from utils import init_weights, init_criterion
from datasets import SegDataset, load_image
from preprocess import Transformation
from EvalAndLoss import *
from predict import valid_seg

# 超参数的设置
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=4, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--save_root', type=str, default='output/seg', help='loss path to save')
parser.add_argument('--trans_name', type=str, default='segmentation', help='chooes transformation type (cyclegan or segmentation)')
parser.add_argument('--init_type', type=str, default='normal',help='initial weight type , inlucde normal,xavier,kaiming')
parser.add_argument('--criterion', type=str, default='crossentropy',help='loss function, include crossentropy,diceloss,focalloss')
parser.add_argument('--model', type=str, default='unet', help='model choosed to segmentation')
parser.add_argument('--histogram_match', type=bool, default=False, help='do histogram match or not')
opt = parser.parse_args()
print(opt)

# Create dir
if os.path.exists(opt.save_root):
    pass
else:
    os.mkdir(opt.save_root)

# Networks
segnet = 0
if opt.model == 'unet':
    segnet = Unet(opt.input_nc, opt.output_nc)
elif opt.model == 'munet':
    segnet = MUnet(opt.input_nc, opt.output_nc)
else:
    raise RuntimeError('no such model:%s'%(opt.model))
segnet.cuda()

# Initial Weights
init_weights(net=segnet,init_type=opt.init_type)

# Lossess
criterion = init_criterion(init_type=opt.criterion)

# Optimizers & LR schedulers
optimizer = torch.optim.Adam(params=segnet.parameters(), lr=opt.lr,betas=(0.9,0.99))
lr_munet = torch.optim.lr_scheduler.StepLR(optimizer,50,gamma=0.1,last_epoch=-1)

# Data Transforms
transforms_ = Transformation(opt).get()
train_trans = transforms_['train']
valid_trans = transforms_['valid']

# Get require data
types = ['C0LGE','LGE','T2LGE']
need_data = load_image(str=types,paired_label=True,histogram_match=True)
image = need_data['image']
label = need_data['label']

# Split dataset(just simply split to train|valid[8:2])
state = np.random.get_state()
np.random.shuffle(image)
np.random.set_state(state)
np.random.shuffle(label)
index = int(len(image) * 0.8)
train_data = {'image':image[:index],'label':label[:index]}
valid_data = {'image':image[index:],'label':label[index:]}

# Load Dataset
train_dataloader = DataLoader(SegDataset(transforms_=train_trans,image=train_data['image'],
                                label=train_data['label'],mode='train'),
                                batch_size=opt.batchSize,shuffle=True,num_workers=opt.n_cpu)
valid_dataloader = DataLoader(SegDataset(transforms_=valid_trans, image=valid_data['image'],
                                label=valid_data['label'],mode='valid'),
                                batch_size=1,shuffle=False,num_workers=opt.n_cpu)                                

# Logger to save info
loss_save = os.path.join(opt.save_root,'loss.csv')
logger = Seglogger(loss_save, opt.n_epochs, len(train_dataloader))

###################################
###### Training & Validation ######
best_dice = 0
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_dataloader):
        segnet.train()
        # [N,1,H,W]
        image = batch['image'].cuda()
        # [N,C,H,W]
        target = batch['target'].cuda()
        # model forward
        predict = segnet(image)
        # back forward
        loss = criterion(predict,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # prepare log data
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        train_loss = loss.item()
        valid_loss = 0
        dice = 0
        jaccard = 0
        # if end of one epoch:  validation
        if ((i+1) % len(train_dataloader)) == 0:
            re = valid_seg(model=segnet,dataloader=valid_dataloader,criterion=criterion)
            valid_loss = re['loss']
            mdice = np.mean(re['dice'][:,0])
            mjaccard = np.mean(re['jaccard'][:,0])
            dice = re['dice']
            jaccard = re['jaccard']
            # save model
            if mdice > best_dice:
                best_dice = mdice
                torch.save(segnet.state_dict(),os.path.join(opt.save_root,'best_dice.pth'))
        # save info
        logger.log({'train_loss':train_loss, 'valid_loss':valid_loss, 'lr':lr,
                'Dice':dice, 'Jaccard':jaccard})
    # step lr rate
    lr_munet.step()

            