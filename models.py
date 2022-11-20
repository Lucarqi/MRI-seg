import torch.nn as nn
import torch
import torch.nn.functional as F

##################################################
#  CycleGAN
##################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        return x
        # Average pooling and flatten
        #return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

###################################################
#   UNet and Modified UNet
###################################################
# convOnce includes {Conv2d, BN , ReLU}
class convOnce(nn.Module):
    def __init__(self, input_c, output_c, kernelsize=3, stride=1, padding=1):
        super(convOnce,self).__init__()
        self.models = nn.Sequential(
            nn.Conv2d(input_c,output_c,kernelsize,stride,padding),
            nn.BatchNorm2d(output_c),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.models(x)

# ConvTwice inludes {(dorp) Conv2d BN ReLU Conv2d BN ReLU}
class convTwice(nn.Module):
    def __init__(self, input_c, output_c, is_drop = False, kernelsize=3, stride=1, padding=1):
        super(convTwice,self).__init__()
        models = []
        if is_drop:
            models += [nn.Dropout()]
        models += [
            convOnce(input_c,output_c, kernelsize, stride, padding),
            convOnce(output_c,output_c, kernelsize, stride, padding),
        ]
        self.models = nn.Sequential(*models)
    
    def forward(self,x):
        x = self.models(x)
        return x


# upsamping class {ConvTrans (ConvTwice)}
class UnetUp(nn.Module):
    def __init__(self, pre_c ,input_c ,add_conv = False, is_drop = False,kernel_size = 4, stide = 2 ,classes = 4, is_scale=True):
        super(UnetUp, self).__init__()
        convin = pre_c + pre_c
        transout = pre_c
        if is_scale: 
            convin = pre_c + classes
            transout = classes
        self.convtwice = convTwice(convin, pre_c, is_drop)
        self.add_conv = add_conv
        models = []
        models += [
            nn.ConvTranspose2d(input_c, transout, kernel_size, stide, padding=1),
        ]
        self.models = nn.Sequential(*models)

    def forward(self, x, pre):
        out1 = self.models(x)
        out = torch.cat([pre, out1],1)
        # ConvTrans + ConvTwice
        if self.add_conv:
            return self.convtwice(out)
        # just ConvTrans
        else:
            return out

# modified U-Net to segmentation
class MUnet(nn.Module):
    '''
    Modified Unet Segmentation Network based on http://arxiv.org/abs/1909.01182
    '''
    def __init__(self, input_c, output_c, classes = 4):
        super(MUnet,self).__init__()
        filers = [64, 128, 256, 512, 1024]

        # downsampling
        # 1 -> 64
        self.conv1 = convTwice(input_c, filers[0], is_drop=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # 64 -> 128
        self.conv2 = convTwice(filers[0], filers[1], is_drop=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # 128 -> 256
        self.conv3 = convTwice(filers[1], filers[2], is_drop=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # 256 -> 512
        self.conv4 = convTwice(filers[2], filers[3], is_drop=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        # 512 -> 1024
        self.bottomneck = convTwice(filers[3],filers[4], is_drop=True)

        # upsampling
        # 1024 -> 512
        self.unetup1 = UnetUp(filers[3], filers[4], add_conv=True, is_drop=True)
        # 512 -> 256
        self.unetup2 = UnetUp(filers[2], filers[3], add_conv=True, is_drop=True)
        self.deepvision1 = convOnce(filers[2], classes)
        self.convtrans1 = nn.ConvTranspose2d(classes, classes, kernel_size=4, stride=2, padding=1)
        # 256 -> 128
        self.unetup3 = UnetUp(filers[1], filers[2], add_conv=True, is_drop=True)
        self.deepvision2 = convOnce(filers[1], classes)
        self.convtrans2 = nn.ConvTranspose2d(classes, classes, kernel_size=4, stride=2, padding=1)
        # 128 -> 64
        self.unetup4 = UnetUp(filers[0], filers[1], add_conv=False)
        self.conv = convOnce(filers[0]+classes, filers[0])
        self.deepvision3 = convOnce(filers[0], classes)

        # out layers
        self.out = nn.Softmax(dim=1)

    def forward(self,x):
        # 1 -> 64
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        # 64 -> 128
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        # 128 -> 256
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        # 256 -> 512
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        # 512 -> 1024
        bottlom = self.bottomneck(maxpool4)
        # 1024 -> 512
        up1 = self.unetup1(bottlom, conv4)
        # 512 -> 256 with deepvision1
        up2 = self.unetup2(up1, conv3)
        deepvision1 = self.deepvision1(up2)
        # 256 -> 128 with deepvision2
        up3 = self.unetup3(up2, conv2)
        deepvision2 = self.deepvision2(up3) + self.convtrans1(deepvision1)
        # 128 -> 64
        up4 = self.unetup4(up3, conv1)
        conv = self.conv(up4)
        deepvision3 = self.deepvision3(conv) + self.convtrans2(deepvision2)

        # final out (no need softmax)
        #out = self.out(deepvision3)
        return deepvision3

# normal unet
class Unet(nn.Module):
    def __init__(self,inc,ouc,classes=4):
        super(Unet,self).__init__()
        # total 4 layer downsampling or upsampling
        # downsampling
        # 1->64
        self.down1 = convTwice(inc,64)
        self.maxplooing1 = nn.MaxPool2d(kernel_size=2)
        # 64->128
        self.down2 = convTwice(64,128)
        self.maxplooing2 = nn.MaxPool2d(kernel_size=2)
        # 128->256
        self.down3 = convTwice(128,256)
        self.maxplooing3 = nn.MaxPool2d(kernel_size=2)
        # 256->512
        self.down4 = convTwice(256,512)
        self.maxplooing4 = nn.MaxPool2d(kernel_size=2)
        # 512->1024
        self.bottomneck = convTwice(512,1024)

        # upsampling
        # 1024->512
        self.unetup1 = UnetUp(512,1024,add_conv=True,is_scale=False)
        # 512->256
        self.unetup2 = UnetUp(256,512,add_conv=True,is_scale=False)
        # 256->128
        self.unetup3 = UnetUp(128,256,add_conv=True,is_scale=False)
        # 128->64
        self.unetup4 = UnetUp(64,128,add_conv=True,is_scale=False)

        # outlayers
        self.conv1 = nn.Conv2d(64,4,kernel_size=1)
    def forward(self,x):
        conv1 = self.down1(x)
        maxpooling1 = self.maxplooing1(conv1)
        conv2 = self.down2(maxpooling1)
        maxpooling2 = self.maxplooing2(conv2)
        conv3 = self.down3(maxpooling2)
        maxpooling3 = self.maxplooing3(conv3)
        conv4 = self.down4(maxpooling3)
        maxpooling4 = self.maxplooing4(conv4)

        bottom = self.bottomneck(maxpooling4)

        up1 = self.unetup1(bottom,conv4)
        up2 = self.unetup2(up1,conv3)
        up3 = self.unetup3(up2,conv2)
        up4 = self.unetup4(up3,conv1)

        out = self.conv1(up4)
        return out

######################################################
#   Attention Unet
######################################################
# conv twice
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionUnet(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(AttentionUnet,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        # attention 
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        # concate
        d5 = torch.cat((x4,d5),dim=1)
        # do conv        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # scale two times
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        # Wg weight
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        # Wx weight
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)


        return x*psi