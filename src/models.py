import numpy as np 
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class ResnetGenerator(nn.Module):

    def __init__(self,n_blocks=6,condition=None):
        super(ResnetGenerator,self).__init__()

        self.n_blocks = n_blocks
        self.condition = condition
        self.encoder = nn.Sequential(
                    nn.Conv2d(5,32,kernel_size=7,padding=3),
                    nn.LeakyReLU(.2),
                    nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
                    nn.LeakyReLU(.2),
                    nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
                    nn.LeakyReLU(.2)
            )

        self.decoder1 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1)
        self.decoder2 = nn.LeakyReLU(.2)
        self.decoder3 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1)
        self.decoder4 = nn.LeakyReLU(.2)
        self.decoder5 = nn.Conv2d(32,3,kernel_size=7,padding=3)
        self.decoder6 = nn.Tanh()

    def res_block(self,x,out_channels=128,filter_width=3):
        """Residual unit with 2 sub layers"""
        orig_x = x
        in_channels = x.shape[1]
        layers = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=filter_width,padding=(filter_width-1)//2),
                nn.LeakyReLU(.2),
                nn.Conv2d(out_channels,out_channels,kernel_size=filter_width,padding=(filter_width-1)//2)
            ).to('cuda')
        #print()
        x = layers(x)
        # if in_channels != out_channels:
        #     orig_x = mypad(orig_x,3,(out_channels-in_channels)//2)
        x += orig_x

        return x

    def mypad(self,x,num_pad):
        """Pads along axis 3 with (num_pad,num_pad)"""
        x = x.to('cpu').numpy()
        npad = ((0,0),(0,0),(num_pad,num_pad))
        x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
        x = torch.from_numpy(x).to('cuda')
        return x

    def forward(self,x):
        #Concatenate
        if self.condition is not None:
            x = torch.cat((x,self.condition),1)

        #Encoder Part
        output = self.encoder(x)

        #Residual part for 6 iterations
        #print(output.shape)
        for i in range(self.n_blocks):
            output = self.res_block(output)

        #Decoder part
        output = self.decoder1(output,output_size=(64,64))
        output = self.decoder2(output)
        output = self.decoder3(output,output_size=(128,128))
        output = self.decoder4(output)
        output = self.decoder5(output)
        output = self.decoder6(output)

        return output 

class PatchDiscriminator(nn.Module):
    """Patch Discriminator which maps input 128X128 image to 14X14 output"""
    def __init__(self,condition=None):
        super(PatchDiscriminator,self).__init__()

        self.condition = condition

        self.layer1 = nn.Sequential(
                    nn.Conv2d(3,64,kernel_size=4,stride=1,padding=(2,1)),
                    nn.LeakyReLU(.2)
            )
        self.layer2 = nn.Sequential(
                    nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
                    nn.LeakyReLU(.2),
                    nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
                    nn.LeakyReLU(.2),
                    nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
                    nn.LeakyReLU(.2),
                    nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1)
            )
    def forward(self,x):
        x = self.layer1(x)

        if self.condition is not None: 
            x = torch.cat((x,self.condition),1)

        output = self.layer2(x)
        return output

class IPAlex(nn.Module):
    """Identity preserving module based on Alexnet trained on Image net"""
    def __init__(self, num_classes=1000):
        super(IPAlex, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.feature1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.features1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def ifeatures(self,x):
        """Returns the activations of conv5 layer for identity preserving features"""
        return self.features(x)