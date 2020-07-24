import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import progressbar
from torchvision import transforms, models
from torchvision.models import alexnet
from torch import nn, optim
from torch.nn import functional as F
import pickle
import progressbar
import argparse
import sys
import os
import PIL
import math
from datetime import datetime 

sys.path.append('/content/drive/My Drive/courseproject-mordor_12/')
print(os.getcwd())
sys.path.append('utils/')
from myDatasets import IMDB_Face_Both
sys.path.append('../src')
#print(os.getcwd())
from models import *
#sys.path.append('../')

TIME_NOW = datetime.now().isoformat()
logger = 'logger.txt'

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
args = parser.parse_args()
checkpoint_path = '.'
#----------PARAMETERS----------------------------------------------------------#

lr = 0.001
weight_decay = 0.0005
momentum = 0.9
train_batch_size = 64
test_batch_size = 64
epochs = 100

lambda1 = 75
lambda2 = 5*math.e-5
lambda3 = 20

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

#------------------------------------------------------------------------------#

def get_training_dataloader():
    """Returns training dataloader"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(), 
        #transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = IMDB_Face_Both(train = True, transform = train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False)

    return trainloader

def get_test_dataloader():
    """Returns test dataloader"""
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    test_dataset = IMDB_Face_Both(train = False, transform = test_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return testloader

trainloader = get_training_dataloader()
testloader = get_test_dataloader()
#------------------------------------------------------------------------------#
# Nets
netD = PatchDiscriminator().to(device)
netG = ResnetGenerator().to(device)
idnet = IPAlex().to(device)
agenet = alexnet(pretrained=False)
agenet = nn.Sequential(
        agenet,
        nn.LeakyReLU(),
        nn.Linear(1000,2),
        nn.LeakyReLU(),
        nn.Softmax(dim = 1)
    ).to(device)

#Load weights
idnet.load_state_dict(torch.load('src/identity-alexnet.pth',map_location=torch.device(device)))

# Load agenet state dict
filename = '/content/drive/My Drive/courseproject-mordor_12/modules/age-detection/checkpoints/checkpoint.pth.tar'
if os.path.isfile(filename):
  checkpoint = torch.load(filename)
  agenet.load_state_dict(checkpoint['state_dict'])

# Make agenet and idnet untrainable
for param in agenet.parameters():
  param.requires_grad = False
for param in idnet.parameters():
  param.requires_grad = False


#SGD Optimizers
optimizerD = optim.SGD(netD.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizerG = optim.SGD(netG.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

#Schedulers
schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size = 15, gamma=0.1)
schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size = 15, gamma=0.1)

#Loss Criterion
criterion = nn.MSELoss().to(device)
age_criterion = nn.CrossEntropyLoss().to(device)

#Global Loss variables
global G_losses
global D_losses
global batch_num
global epoch_lst
global D_X
global D_Gz1
global D_Gz2
epoch_lst = []
G_losses = []
D_losses,D_X,D_Gz1,D_Gz2 = [],[],[],[]
batch_num = 0

#------------------------------------------------------------------------------#
#Saving and Loading the  model
def save_pickle(e, batch_num, name = 'data.pickle'):
    dic = {}
    dic['Glosses'] = G_losses
    dic['Dlosses'] = D_losses
    dic['epoch'] = epoch_lst
    dic['batch_num'] = batch_num_lst
    dic['accuracies'] = accuracies
    with open(checkpoint_path + name, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_checkpoint(e, batch_num, filename=checkpoint_path + 'checkpoint.pth.tar'):
    state = {
            'epoch': e + 1,
            'batch_num': batch_num,
            'gen_dict': netG.state_dict(),
            'disc_dict': netD.state_dict(), 
            'optimizerD' : optimizerD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'schedulerD' : schedulerD.state_dict(),
            'schedulerG' : schedulerG.state_dict(),
        }

    torch.save(state, filename)
    print('Saving the model after {} epochs'.format(e+1))
    # save_pickle(e, batch_num, 'checkpoint.pickle')

def load_checkpoint(filename=checkpoint_path + 'checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        global start_epoch
        start_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['gen_dict'])
        netD.load_state_dict(checkpoint['disc_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    # if os.path.isfile(checkpoint_path + 'checkpoint.pickle'):
    #     with open(checkpoint_path + 'checkpoint.pickle', 'rb') as handle:
    #         dic = pickle.load(handle)
    #         global memory_losses
    #         global memory_accuracies
    #         global memory_epoch_lst
    #         global memory_batch_num_lst
    #         memory_losses = dic['losses']
    #         memory_epoch_lst = dic['epoch']
    #         memory_batch_num_lst = dic['batch_num']
    #         memory_accuracies = dic['accuracies']
    #         print('Loaded data\n\n')
    # else:
    #     print("=> no checkpoint found at '{}'".format(checkpoint_path + 'checkpoint.pickle'))

if args.resume:
    load_checkpoint()
    # losses = memory_losses
    # epoch_lst = memory_epoch_lst
    # batch_num_lst = memory_batch_num_lst
    # accuracies = memory_accuracies
        
    print('Resuming Training\n')
else:
    print('Starting Training\n')
    start_epoch = 0
#------------------------------------------------------------------------------#
def train(epoch):
    bar = progressbar.ProgressBar(maxval=len(trainloader), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bari = 0
    for batch_num, (images,_,labels) in enumerate(trainloader,0):
        #print("==========>",labels)
        # netD.zero_grad() 
        optimizerD.zero_grad()
        images =  images.to(device)
        labels = labels.to(device)

        #Discriminator on real image
        real_image = images.to(device)
        output = netD(real_image)
        D_x = output.mean()
        real_label = torch.ones(output.shape,device=device)
        Ld_real = criterion(output,real_label)

        #Discriminator on fake image
        n,_,h,w = images.shape
        condition = torch.zeros(n,2,h,w).to(device)
        for k,j in enumerate(labels):
            condition[k,1-j,:,:]=1

        inputG = torch.cat((real_image,condition),1).to(device)
        fake_image = netG(inputG)
        output = netD(fake_image.detach())
        D_G_z1 = output.mean()
        fake_label = torch.zeros(output.shape,device=device)
        Ld_fake = criterion(output,fake_label)
        errD =  .5*(Ld_real + Ld_fake)
        #print(errD.requires_grad)
        errD.backward()
        optimizerD.step()

        # Put idnet and agenet in eval mode
        idnet.eval()
        agenet.eval()

        #Generator

        #netG.zero_grad()
        optimizerG.zero_grad()
        
        output = netD(fake_image)
        #Loss due to the discriminator
        Lg = criterion(output,real_label)
        D_G_z2 = output.mean()

        fake_features = idnet.ifeatures(fake_image)
        real_features = idnet.ifeatures(real_image)

        #Identity preserving module loss
        Lid = criterion(fake_features,real_features)

        #Age enforcing loss
        pred_age = agenet(fake_image) 
        Lage = age_criterion(pred_age,1-labels)
        errG = lambda1*.5*Lg + lambda2*Lid + lambda3*Lage
        errG.backward()
        optimizerG.step()

        if batch_num % 50 == 0:
            image_inpt = np.transpose(images[0].cpu().detach().numpy(),[1,2,0])
            plt.figure()
            plt.imshow(image_inpt)
            plt.savefig('src/fundo.png')
            plt.close('all')
            print('\n[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, batch_num, len(trainloader),
                     errD.item(), errG.item(), D_x.item(), D_G_z1.item(), D_G_z2.item()))

            # with open('logger.txt','w') as f1:
            #     f1.write('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(trainloader),errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if batch_num % 200 == 0:
            image_inpt = np.transpose(images[0].cpu().detach().numpy(),[1,2,0])
            image_otpt = np.transpose(fake_image[0].cpu().detach().numpy(),[1,2,0])
            plt.figure()
            plt.subplot(121)
            plt.imshow(image_inpt)
            plt.subplot(122)
            plt.imshow(image_otpt)
            plt.savefig('src/Output.png')
            plt.close('all')

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        D_X.append(D_x.item())
        D_Gz1.append(D_G_z1.item())
        D_Gz2.append(D_G_z2.item())

        bari += 1
        bar.update(bari)
    bar.finish()

#------------------------------------------------------------------------------#
#Training thee GAN
print("Starting Training Loop...")
for e in range(epochs):
    save_checkpoint(e, batch_num)
    e += start_epoch
    if e >= epochs:
        break
    train(e)
    schedulerG.step()
    schedulerD.step()
    
    print('\n')    
    # save_checkpoint(e, batch_num, model, optimizer)

