import random
import time
import datetime
import sys

from models import Unet,MUnet
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from EvalAndLoss import CrossEntropyLoss,FocalLoss,DiceLoss

def reverse_centercrop(image,size):
    '''
    Padding 0 around given image to given size
    Input:
        `image`: a numpy of [h,w]
        `size`: size of padded image
    Output:
        a numpy of padding image
    '''
    assert size > image.shape[0]
    assert (size + image.shape[0]) % 2 == 0
    out = np.zeros((size,size))
    xmin = int((size - image.shape[0]) / 2)
    xmax = xmin + image.shape[0]
    ymin = xmin
    ymax = xmax
    out[xmin:xmax,ymin:ymax] = image
    return out 

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

def mask2onehot(mask, label):
    '''
    convert label to onehot mask
    input:
        mask : a numpy of mask [h,w]
        label : a numpy of remark
    output :
        a tensor [4, H ,W] for 4 classes {backgroud LV myo RV}
    '''
    # expand dims to [H W 1]
    mask = np.expand_dims(mask, axis=2)
    seg_map = []
    for i in label:
        equality = np.equal(i, mask)
        class_map = np.all(equality, axis=-1)
        seg_map.append(class_map)
    # get seg_map [H,W,4]
    seg_map = np.stack(seg_map, axis=-1).astype(np.float32)
    # convert to tensor
    out = torch.tensor(seg_map).permute(2,0,1)
    return out

def onehot2mask(onehot, label):
    '''
    convert one hot to mask
    input:
        `onehot` : a tensor of one hot [4, H ,W]
        `label` : remark of mask
    output:
        a numpy of mask [H,W], dtype is numpy float32
    '''
    # convert to numpy
    onehot = onehot.numpy()
    x = np.argmax(onehot, axis=0)
    color_codes = np.array(label).squeeze(axis=-1)
    for i in range(0,4):
        x[x==i] = color_codes[i] 
    return x

def drawhistogram(data):
    '''
    draw histogram of input numpy
    input:
        data - a numpy dimension of [h,w]
    '''
    img = data.ravel()
    plt.hist(img,color='red',bins=200)
    plt.show()

def denormalization(tensor):
    input = tensor.cpu().float().numpy()
    mean = 0.5
    std = 0.5

    input = input * std + mean
    input = input * 255.0
    return np.squeeze(input,axis=0)

def tensor2nii(tensor,info):
    '''
    save cyclegan output
    from tensor float [1,1,h,w] range of [-1,1]
    to numpy float .nii.gz file range of [0,255]
    
    input :
        tensor , info (as file name)
    output: 
        None ,but create a file named "info".nii.gz
    '''
    de_img = denormalization(tensor)
    de_img = sitk.GetImageFromArray(de_img)
    # need to modify image save path
    sitk.WriteImage(de_img,'datasets/train/fake_lge/%s.nii'%(info))

# MinMaxScaler and Normalization
def minmax_normal(input):
    '''
    input :
        tensor [1,h,w]
    return :
        minmaxscaler and normalization tensor, range of [-1,1]
    '''
    scaler = (input - torch.min(input))/(torch.max(input) - torch.min(input)) # convert to [0,1]
    normal = (scaler - 0.5) / 0.5 # convert to [-1,1]
    return normal

def normal_normal(input):
    '''
    input :
        tensor [1,h,w]
    return :
        normal normalization tensor, which means = 0.5, std = 0.5
    '''
    normal = (input - torch.mean(input,dim=(1,2),keepdim=True)) / (torch.std(input,dim=(1,2),keepdim=True))
    out = normal * 0.5 + 0.5
    return out

# ??????????????????
class Logger():
    def __init__(self, n_epochs, batches_epoch, save_root):
        #self.viz = Visdom(env='cycleGAN')
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        self.save_root = save_root

        df = pd.DataFrame(columns=['epoch','loss_G','loss_G_identity','loss_G_GAN',
                            'loss_G_cycle','loss_D'])
        df.to_csv(self.save_root,index=False)
    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        # print loss,runtime etc. in terminal per one batch
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        '''
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.images(denormalization(tensor.data), opts={'title':image_name})
            else:
                self.viz.images(denormalization(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})
        '''
        # Draw loss per epoch in web browser
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            data_ = [self.epoch]
            for loss_name, loss in self.losses.items():
                data_.append(loss/self.batch)
                if loss_name not in self.loss_windows:
                    #self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                    pass
                else:
                    #self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                    pass
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            # save in disk
            data = pd.DataFrame([data_])
            data.to_csv(self.save_root, mode='a', header=False, index=False)

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        
# ??????Generator?????????????????????0.5??????????????????????????????????????????????????????
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

# ??????????????????????????????epoch??????opt.decay_start_epoch
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

# ???????????????
def init_weights(net,init_type='normal'):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=0.02)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, nonlinearity='relu',mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m,'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_criterion(init_type='crossentropy'):
    '''
    choose criterion
    input:
        `init_type` -- criterion type
    '''
    if init_type == 'crossentropy':
        return CrossEntropyLoss(reduction='mean')
    if init_type == 'focalloss':
        return FocalLoss(reduction='mean')
    if init_type == 'diceloss':
        return DiceLoss(reduction='mean')
    else:
        raise RuntimeError('no such loss function')

# Segmentation ????????????
class Seglogger():
    def __init__(self,save_root,total_epoch, batch_epoch):
        #self.writer = SummaryWriter(opt.save_root)
        self.prev_time = time.time()
        self.train_loss = 0
        self.mean_time = 0
        self.save_root = save_root
        # current batch
        self.batch = 1
        # current epoch
        self.epoch = 1
        # total epoch
        self.total_epoch = total_epoch
        # total batch in one epoch
        self.batch_epoch = batch_epoch
        # save info at end of one epoch training
        train = pd.DataFrame(columns=['epoch','lr','train_loss','valid_loss',
                                        'Dice','Dice_Myo','Dice_LV','Dice_RV',
                                        'Jaccard','Jaccard_Myo','Jaccard_LV','Jaccard_RV'])
        train.to_csv(self.save_root,index=False)
    def log(self,data=None):
        # save time for rest time compute
        self.mean_time += time.time() - self.prev_time
        self.prev_time = time.time()
        # print loss,runtime etc. in terminal per one batch
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.total_epoch, self.batch, self.batch_epoch))
        self.train_loss += data['train_loss']
        lr = data['lr']
        sys.stdout.write('%s: %.4f | lr: %.8f | ' % ('train_loss', self.train_loss / self.batch, lr))
        batch_done = self.batch_epoch*(self.epoch - 1) + self.batch
        batch_left = self.batch_epoch*(self.total_epoch - self.epoch) + self.batch_epoch - self.batch
        sys.stdout.write('ETA %s' %(datetime.timedelta(seconds=batch_left*self.mean_time/batch_done)))
        # save info at each epoch ends
        if(self.batch % self.batch_epoch) == 0:
            valid_loss = data['valid_loss']
            dice = data['Dice']
            jaccard = data['Jaccard']
            mdice = np.mean(dice[:,0])
            mjaccard = np.mean(jaccard[:,0])
            sys.stdout.write('\n %s: %.4f | %s: %.4f | %s: %.4f \n' % ('valid_loss',valid_loss,'Dice',mdice,'Jaccard',mjaccard))
            save_data = [self.epoch,lr,self.train_loss/self.batch,valid_loss,
                        mdice, dice[0,0],dice[1,0],dice[2,0],
                        mjaccard, jaccard[0,0],jaccard[1,0],jaccard[2,0]]
            df = pd.DataFrame([save_data])
            df.to_csv(self.save_root, header=False, index=False, mode='a')
            self.train_loss = 0
            self.epoch += 1
            self.batch = 1
        else:
            self.batch += 1