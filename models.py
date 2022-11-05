from turtle import forward, up
import torch.nn as nn
import torch
import torch.nn.functional as F

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
    def __init__(self, pre_c ,input_c ,add_conv = False, is_drop = False,kernel_size = 4, stide = 2 ,classes = 4):
        super(UnetUp, self).__init__()
        self.convtwice = convTwice(pre_c+classes, pre_c, is_drop)
        self.add_conv = add_conv
        models = []
        models += [
            nn.ConvTranspose2d(input_c, classes, kernel_size, stide, padding=1),
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