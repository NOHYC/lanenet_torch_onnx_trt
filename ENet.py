
'''
ENet [Reference: https://arxiv.org/pdf/1606.02147.pdf]
'''

import torch
from torch.nn import init
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class InitialBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InitialBlock, self).__init__()
        self.input_channel = in_ch
        self.conv_channel = out_ch - in_ch

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch - in_ch, kernel_size = 3, stride = 2, padding=1),
            nn.BatchNorm2d(out_ch - in_ch),
            nn.PReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        conv_branch = self.conv(x)
        maxp_branch = self.maxpool(x)
        return torch.cat([conv_branch, maxp_branch], 1)
        
class BottleneckModule_Downsampling(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 1, dropout_prob = 0,img_size = None):
        super(BottleneckModule_Downsampling, self).__init__()
        self.input_channel = in_ch   
        self.activate = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 2, stride = 2),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Dropout2d(p=dropout_prob)
        )
    def forward(self, x):
        conv_branch = self.conv(x)
        maxp_branch = self.maxpool(x)
        bs, conv_ch, h, w = conv_branch.size()
        maxp_ch = maxp_branch.size()[1]
        #padding = torch.zeros(bs, conv_ch - maxp_ch, h, w)
        #maxp_branch = torch.cat([maxp_branch, padding], 1)
        padding = torch.zeros(bs, conv_ch - maxp_ch, h, w).to(DEVICE)
        maxp_branch = torch.cat([maxp_branch, padding], 1).to(DEVICE)
        output = maxp_branch + conv_branch

        return self.activate(output)

class BottleneckModule_Upsampling4_0(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 1,dropout_prob = 0):
        super(BottleneckModule_Upsampling4_0, self).__init__()
        self.input_channel = in_ch
        self.activate = nn.PReLU()

        self.con = nn.Conv2d(in_ch, out_ch, kernel_size = 1)
        self.batch = nn.BatchNorm2d(out_ch)
        self.ups = nn.Upsample(size=(64,128), mode='nearest')

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Dropout2d(p=dropout_prob)
        )
    def forward(self, x):
        conv_branch = self.conv(x)
        #maxunp_branch = self.maxunpool(x)
        x = self.con(x)
        x = self.batch(x)
        maxunp_branch  = self.ups(x)

        output = maxunp_branch + conv_branch
        return self.activate(output)        

class BottleneckModule_Upsampling5_0(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 1, dropout_prob = 0):
        super(BottleneckModule_Upsampling5_0, self).__init__()
        self.input_channel = in_ch
        self.activate = nn.PReLU()

        self.con = nn.Conv2d(in_ch, out_ch, kernel_size = 1)
        self.batch = nn.BatchNorm2d(out_ch)
        self.ups = nn.Upsample(size=(128,256), mode='nearest')


        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Dropout2d(p=dropout_prob)
        )



    def forward(self, x):
        conv_branch = self.conv(x)
        x = self.con(x)
        x = self.batch(x)
        maxunp_branch  = self.ups(x)



        
        output = maxunp_branch + conv_branch
        return self.activate(output)
class BottleneckModule_Regular(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 1, dropout_prob = 0):
        super(BottleneckModule_Regular, self).__init__()
        self.input_channel = in_ch
        self.activate = nn.PReLU()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Dropout2d(p=dropout_prob)
        )
    def forward(self, x):
        output = self.conv(x) + x
        return self.activate(output)

class BottleneckModule_Asymmetric(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 1,asymmetric = 5, dropout_prob = 0):
        super(BottleneckModule_Asymmetric, self).__init__()
        self.input_channel = in_ch
        self.activate = nn.PReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, (asymmetric, 1), stride=1, padding=(padding, 0)),
            nn.Conv2d(out_ch, out_ch, (1, asymmetric), stride=1, padding=(0, padding)),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Dropout2d(p=dropout_prob)
        )
    def forward(self, x):
        output = self.conv(x) + x
        return self.activate(output)


class BottleneckModule_Dilated(nn.Module):
    def __init__(self, in_ch, out_ch, padding = 1, dilated = 0, dropout_prob = 0):
        super(BottleneckModule_Dilated, self).__init__()
        self.input_channel = in_ch
        self.activate = nn.PReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride=1, padding=padding, dilation=dilated),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size = 1),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
            nn.Dropout2d(p=dropout_prob)
            )
    def forward(self, x):
        output = self.conv(x) + x
        return self.activate(output)

class ENet_Encoder(nn.Module):
    
    def __init__(self, in_ch=3, out_ch=1):
        super(ENet_Encoder, self).__init__()

        self.initial_block = InitialBlock(in_ch, 16)

        self.bottleneck1_0 = BottleneckModule_Downsampling(16, 64,  padding = 1, dropout_prob = 0.1)
        self.bottleneck1_1 = BottleneckModule_Regular(64, 64, padding = 1, dropout_prob = 0.1)
        self.bottleneck1_2 = BottleneckModule_Regular(64, 64, padding = 1, dropout_prob = 0.1)
        self.bottleneck1_3 = BottleneckModule_Regular(64, 64, padding = 1, dropout_prob = 0.1)
        self.bottleneck1_4 = BottleneckModule_Regular(64, 64, padding = 1, dropout_prob = 0.1)

        self.bottleneck2_0 = BottleneckModule_Downsampling(64, 128, padding = 1, dropout_prob = 0.1)
        self.bottleneck2_1 = BottleneckModule_Regular(128, 128, padding = 1, dropout_prob = 0.1)
        self.bottleneck2_2 = BottleneckModule_Dilated(128, 128, padding = 2, dilated = 2, dropout_prob = 0.1)
        self.bottleneck2_3 = BottleneckModule_Asymmetric(128, 128, padding = 2, asymmetric=5, dropout_prob = 0.1)
        self.bottleneck2_4 = BottleneckModule_Dilated(128, 128, padding = 4, dilated = 4, dropout_prob = 0.1)
        self.bottleneck2_5 = BottleneckModule_Regular(128, 128,  padding = 1, dropout_prob = 0.1)
        self.bottleneck2_6 = BottleneckModule_Dilated(128, 128, padding = 8, dilated = 8, dropout_prob = 0.1)
        self.bottleneck2_7 = BottleneckModule_Asymmetric(128, 128,  padding = 2, asymmetric=5, dropout_prob = 0.1)
        self.bottleneck2_8 = BottleneckModule_Dilated(128, 128, padding = 16, dilated = 16, dropout_prob = 0.1)

        self.bottleneck3_0 = BottleneckModule_Regular(128, 128, padding = 1, dropout_prob = 0.1)
        self.bottleneck3_1 = BottleneckModule_Dilated(128, 128, padding = 2, dilated = 2, dropout_prob = 0.1)
        self.bottleneck3_2 = BottleneckModule_Asymmetric(128, 128, padding = 2, asymmetric=5, dropout_prob = 0.1)
        self.bottleneck3_3 = BottleneckModule_Dilated(128, 128, padding = 4, dilated = 4, dropout_prob = 0.1)
        self.bottleneck3_4 = BottleneckModule_Regular(128, 128, padding = 1, dropout_prob = 0.1)
        self.bottleneck3_5 = BottleneckModule_Dilated(128, 128, padding = 8, dilated = 8, dropout_prob = 0.1)
        self.bottleneck3_6 = BottleneckModule_Asymmetric(128, 128, padding = 2, asymmetric=5, dropout_prob = 0.1)
        self.bottleneck3_7 = BottleneckModule_Dilated(128, 128, padding = 16, dilated = 16, dropout_prob = 0.1)
    
    def forward(self, x):

        x = self.initial_block(x)

        x = self.bottleneck1_0(x)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        x = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        x = self.bottleneck3_0(x)
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)

        return x

class ENet_Decoder(nn.Module):
    
    def __init__(self, out_ch=1):
        super(ENet_Decoder, self).__init__()

        self.bottleneck4_0 = BottleneckModule_Upsampling4_0(128, 64,  padding = 1, dropout_prob = 0.1)
        self.bottleneck4_1 = BottleneckModule_Regular(64, 64, padding = 1, dropout_prob = 0.1)
        self.bottleneck4_2 = BottleneckModule_Regular(64, 64, padding = 1, dropout_prob = 0.1)

        self.bottleneck5_0 = BottleneckModule_Upsampling5_0(64, 16, padding = 1, dropout_prob = 0.1)
        self.bottleneck5_1 = BottleneckModule_Regular(16, 16,  padding = 1, dropout_prob = 0.1)

        self.fullconv = nn.ConvTranspose2d(16, out_ch, kernel_size=2, stride=2)

    def forward(self, x):    
        x = self.bottleneck4_0(x)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        x = self.bottleneck5_0(x)
        x = self.bottleneck5_1(x)

        x = self.fullconv(x)

        return x