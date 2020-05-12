import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class ResnetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=(7,7), padding=3, bias=False), #Unclear about input dim
            #nn.BatchNorm2d(32, ) How to add BatchNorm for (N,H,W,C) to(N,C,H,W)
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(32, 64,kernel_size=(3,3), padding=1, bias=False),
            #BatchNorm2d
            nn.RelU()
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(64, 128,kernel_size=(3,3), padding=1, bias=False),
            #BatchNorm2d
            nn.RelU()
        )
        '''
        ADD Residual Block
        '''
        '''
        ADD deconv blocks
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64),
            #BatchNorm2d
            nn.ReLU()
        )
        self.hidden5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32),
            #BatchNorm2d
            nn.ReLU()
        )
        '''
        self.hidden6 = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=(7,7))
        )
        
    def forward(self,x,labels):
        #Fill

class PatchDiscriminator(nn.Module):
    def __init__(self):
        #Fill
    
    def forward(self, x, labels):
        #Fill