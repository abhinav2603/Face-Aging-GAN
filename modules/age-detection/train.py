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
import argparse
import sys
import os
import PIL

sys.path.append('/content/drive/My Drive/courseproject-mordor_12/utils/')
from myDatasets import IMDB_Face_Both

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--on_colab', action = 'store_true')
args = parser.parse_args()

cuda_enable = True
lr = 0.01
weight_decay = 0.0005
momentum = 0.9
train_batch_size = 64
test_batch_size = 64
epochs = 1000

if not args.on_colab:
    checkpoint_path = 'checkpoints/'
else:
    cuda_enable = True
    checkpoint_path = './checkpoints/'

global losses
global accuracies
global epoch_lst
global batch_num_lst

losses = []
accuracies = []
epoch_lst = []
batch_num_lst = []

train_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
test_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

if not args.eval:
    trainset = IMDB_Face_Both(train = True, transform = train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False, num_workers=1)
else:
    valset = IMDB_Face_Both(train = False, transform = test_transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=test_batch_size, shuffle=False, num_workers=1)

model = alexnet(pretrained = True)

model = nn.Sequential(
        model,
        nn.LeakyReLU(),
        nn.Linear(1000,2),
        nn.LeakyReLU(),
        nn.Softmax(dim = 1)
    )
    
if cuda_enable:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150, 250, 350], gamma=0.1)

if cuda_enable:
    criterion = criterion.cuda()

# Define analysis function
def analyse(model, per_class_accuracy_needed = False, ontrain = False):
    test_correct=0
    test_loss = 0
    if ontrain:
        dloader, dset = trainloader, trainset
    else:
        dloader, dset = valloader, valset
    test_total = len(dset)
    bari = 0
    bar = progressbar.ProgressBar(maxval=test_total, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    if per_class_accuracy_needed:
        per_class_accuracy = np.zeros((2, 2))

    with torch.no_grad():
        #perform a prediction on the validation  data 
        for x_test, _, y_test in dloader:
            if cuda_enable:
                x_test, y_test = x_test.cuda(), y_test.cuda()

            model.eval()
            output = model(x_test)
            test_loss += criterion(output, y_test)
            _, prediction = torch.max(output.data, 1)

            test_correct += np.sum(prediction.cpu().numpy() == y_test.cpu().numpy())

            if per_class_accuracy_needed:
                for i in range(prediction.shape[0]):
                    if label_numpy[i] == prediction[i]:
                        per_class_accuracy[label_numpy[i], 1] += 1
                    else:
                        per_class_accuracy[label_numpy[i], 0] += 1
            bari += dloader.batch_size
            bar.update(bari)
            del x_test
            del output
            if cuda_enable:
                torch.cuda.empty_cache()
    
    bar.finish()
    accuracy = test_correct / test_total
    if per_class_accuracy_needed:
        return accuracy, per_class_accuracy    
    return test_loss, accuracy


def save_pickle(e, batch_num, name = 'data.pickle'):
    dic = {}
    dic['losses'] = losses
    dic['epoch'] = epoch_lst
    dic['batch_num'] = batch_num_lst
    dic['accuracies'] = accuracies
    with open(checkpoint_path + name, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_checkpoint(e, batch_num, model, optimizer, filename=checkpoint_path + 'checkpoint.pth.tar'):
    state = {
            'epoch': e + 1,
            'batch_num': batch_num,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }

    torch.save(state, filename)
    print('Saving the model after {} epochs'.format(e+1))
    save_pickle(e, batch_num, 'checkpoint.pickle')

def load_checkpoint(filename=checkpoint_path + 'checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        global start_epoch
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    if os.path.isfile(checkpoint_path + 'checkpoint.pickle'):
        with open(checkpoint_path + 'checkpoint.pickle', 'rb') as handle:
            dic = pickle.load(handle)
            global memory_losses
            global memory_accuracies
            global memory_epoch_lst
            global memory_batch_num_lst
            memory_losses = dic['losses']
            memory_epoch_lst = dic['epoch']
            memory_batch_num_lst = dic['batch_num']
            memory_accuracies = dic['accuracies']
            print('Loaded data\n\n')
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path + 'checkpoint.pickle'))

def print_per_class_accuracy(pca_mat):
    for i in range(pca_mat.shape[0]):
        print('Label {} = {}%'.format(i,100*pca_mat[i,1]/(pca_mat[i,0] + pca_mat[i,1])))

if args.eval:
    load_checkpoint()
    _, A, pca = analyse(model, per_class_accuracy_needed = True, ontrain = False)
    print('\n\nEvaluating on the Validation set.\nAccuracy of the currently saved model is {} %'.format(100 * A))
    print_per_class_accuracy(pca)

    os._exit(0)


if args.resume:
    load_checkpoint()
    losses = memory_losses
    epoch_lst = memory_epoch_lst
    batch_num_lst = memory_batch_num_lst
    accuracies = memory_accuracies
        
    print('Resuming Training\n')
else:
    print('Starting Training\n')
    start_epoch = 0

prev_save = 0
running_loss = 0
strike = 0
bar_finished = True
bar_size = 400
bari = 0
for e in range(epochs):
    e += start_epoch
    if e >= epochs:
        break
    train_loss = 0
    train_correct = 0
    total = 0
    bari = 0

    for batch_num, (images, _, labels) in enumerate(trainloader,1):
        if bar_finished:
            bari = 0
            print('Epoch: {} | Batches {}-{}'.format(e+1, bar_size*(batch_num//bar_size), bar_size*(1+(batch_num//bar_size))))
            bar = progressbar.ProgressBar(maxval=bar_size, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            bar_finished = False
        if cuda_enable:
            images, labels = images.cuda(), labels.cuda()

        # Training pass
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, prediction = torch.max(output, 1)
        
        with open('train_pred.txt', 'w') as f:
            f.write(str('Prediction') + '\n')
            f.write(str(prediction[1]) + '\n')
            f.write(str('Output') + '\n')
            f.write(str(output) + '\n')
            f.write(str('Labels') + '\n')
            f.write(str(labels) + '\n')
        
        total += labels.size(0)
        train_correct += np.sum(prediction.cpu().numpy() == labels.cpu().numpy())

        # if batch_num % 100 == 0 or (prev_save < .70 and batch_num % 50 == 0):
        if batch_num % bar_size == 0:
            bari += 1
            bar.update(bari)
            bar.finish()
            bar_finished = True
            print('\nBatch Number : {}/{}'.format(batch_num, len(trainset)//images.shape[0]), flush = True)
            print('Train Loss : {}'.format(train_loss/batch_num), flush = True)
            print('Train Accuracy : {} %\n'.format(100 * (train_correct/total)), flush = True)
            if train_correct == total:
                strike += 1
            else:
                strike = 0

            if strike == 100:
                break

            with open('manual_stop.txt', 'r') as f:
                if f.readline().strip() == '1':
                    print('Manual Override: Force Halting')
                    break
        else:
            bari += 1
            bar.update(bari)

        del images
        del labels
        if cuda_enable:
            torch.cuda.empty_cache()

    scheduler.step()
    print('\n')    
    save_checkpoint(e, batch_num, model, optimizer)

    if strike == 100:
        print('Model Fully Trained. Interrupting training.')
        break
    with open('notify.txt', 'w') as f:
        f.write('SHUTDOWN OKAY')
        f.flush()

    with open('manual_stop.txt', 'r') as f:
        if f.readline().strip() == '1':
            os.system('echo 0 > manual_stop.txt')
            break