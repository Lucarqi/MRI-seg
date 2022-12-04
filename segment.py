# segmentation train step
import os
import random
import argparse
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn

from models import segment_model
from utils import init_weights, init_criterion ,Seglogger
from datasets import SegDataset, makedatasets
from preprocess import Transformation
from EvalAndLoss import *
from predict import valid_seg

# 超参数的设置
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=4, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--save_root', type=str, default='output/seg', help='loss path to save')
parser.add_argument('--name', type=str, default='segmentation', help='chooes transformation type (cyclegan or segmentation)')
parser.add_argument('--init_type', type=str, default='normal',help='initial weight type , inlucde normal,xavier,kaiming')
parser.add_argument('--criterion', type=str, default='crossentropy',help='loss function, include crossentropy,diceloss,focalloss')
parser.add_argument('--model', type=str, default='aunet', help='model choosed to segmentation, inlucde|unet|munet|aunet')
parser.add_argument('--histogram_match', type=bool, default=False, help='do histogram match or not')
parser.add_argument('--lock',type=bool, default=True, help='lock random seed or not')
opt = parser.parse_args()
print(opt)

# lock random
seed = 2022
def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True #确定性固定
    #torch.backends.cudnn.benchmark = True #False会确定性地选择算法，会降低性能
    torch.manual_seed(seed)
def _init_fn(work_id):
    np.random.seed(int(seed)+work_id)
work_init_fn = None
if opt.lock:
    seed_it(seed)
    work_init_fn = _init_fn

# Create dir
if os.path.exists(opt.save_root):
    pass
else:
    os.mkdir(opt.save_root)

# Networks
segnet = segment_model(opt)
segnet.cuda()

# Initial Weights
init_weights(net=segnet,init_type=opt.init_type)

# Lossess
a = None
if opt.criterion != 'bdloss':
    criterion = init_criterion(init_type=opt.criterion)
else:
    criterion, boundary_loss = CrossEntropyLoss(), BoundaryLoss(idc=[1,2,3])
    a = 0.01
# Optimizers & LR schedulers
#optimizer = torch.optim.SGD(params=segnet.parameters(), lr=opt.lr, momentum=0.9)
optimizer = torch.optim.Adam(params=segnet.parameters(), lr=opt.lr, betas=(0.9,0.99))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,gamma=0.1,step_size=50)

# Data Transforms
transforms_ = Transformation(opt).get()
train_trans = transforms_['train']

# Get require data and validation path
types = ['LGE','C0LGE','T2LGE']
image, label , valid_path = makedatasets(types,lge_valid=False,split=0.2)

# Load Dataset
train_dataloader = DataLoader(SegDataset(transforms_=train_trans,image=image,label=label),
                                batch_size=opt.batchSize,shuffle=True,num_workers=opt.n_cpu)

# Logger to save info
loss_save = os.path.join(opt.save_root,'loss.csv')
logger = Seglogger(loss_save, opt.n_epochs, len(train_dataloader))

###################################
###### Training & Validation ######
best_dice = 0
iters = len(train_dataloader)
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_dataloader):
        segnet.train()
        # [N,1,H,W]
        image = batch['image'].cuda()
        # [N,4,H,W]
        target = batch['target'].cuda()
        dist_map = batch['dist_map'].cuda()
        # model forward
        predict = segnet(image)
        # back forward
        loss = criterion(predict,target)
        if opt.criterion == 'bdloss':
            loss = loss + a * boundary_loss(predict, dist_map)
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
            loss, scores = valid_seg(model=segnet,datapath=valid_path,type=opt.criterion,a=a)
            valid_loss = loss
            mdice = np.mean(scores[1:])
            dice = scores
            dice[0] = mdice
            # save model
            if mdice > best_dice:
                best_dice = mdice
                torch.save(segnet.state_dict(),os.path.join(opt.save_root,'best_dice.pth'))
        # save info
        logger.log({'train_loss':train_loss, 'valid_loss':valid_loss, 'lr':lr,
                'Dice':dice})
    # lr step
    scheduler.step()

            