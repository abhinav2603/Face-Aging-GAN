import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import h5py
import pickle
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import PIL

DATA_FOLDER_PATH = '/content/drive/Shared drives/MordorNirajAbhinav/datasets/'

class IMDB_Face_Young(Dataset):
    """docstring for IMDB_Face"""
    def __init__(self, data_pre_path = DATA_FOLDER_PATH + '10_30/10_30_', transform=lambda x:x, train = True, entire=False):
        super(IMDB_Face_Young, self).__init__()
        self.chunk_opened = None
        self.images = None
        self.ages = None
        self.train = train
        if entire:
            self.datalen = 129452
            self.index_start = 0
        else:
            if train:
                self.datalen = 100000
                self.index_start = 0
            else:
                self.datalen = 29452
                self.index_start = 100000
        self.chunk_size = 5000
        self.data_pre_path = data_pre_path
        self.transform = transform

    def __isIndexOpened(self, index):
        if self.chunk_opened is None:
            return False
        return ((index >= self.chunk_opened * self.chunk_size) and (index < (self.chunk_opened+1) * self.chunk_size))
        
    def __indexToChunk(self, index):
        return index//self.chunk_size
    
    def __load_chunk(self, i):
        i = int(i)
        if i == self.chunk_opened:
            return
        else:
            # print(self.data_pre_path,i)
            file = h5py.File(self.data_pre_path + str(i) + '.h', "r+")
            self.images = np.array(file["/images"]).astype("uint8")
            self.ages = np.array(file["/meta"]).astype("uint8")
            self.chunk_opened = i

    def visualize(self, i):
        i += self.index_start
        im, age = self.__getitem__(i)
        im = im.detach().cpu().numpy()
        plt.figure()
        plt.title(str(age))
        plt.imshow(np.swapaxes(im, 0, 2))
        plt.show()

    def __getitem__(self, i):
        i += self.index_start
        if not self.__isIndexOpened(i):
            self.__load_chunk(self.__indexToChunk(i))
        return self.transform(self.images[i%self.chunk_size]), torch.DoubleTensor([self.ages[i%self.chunk_size]])

    def __len__(self):
        return self.datalen


class IMDB_Face_Old(Dataset):
    """docstring for IMDB_Face"""
    def __init__(self, data_pre_path = DATA_FOLDER_PATH + '50_80/50_80_', transform=lambda x:x, train = True, entire=False):
        super(IMDB_Face_Old, self).__init__()
        self.chunk_opened = None
        self.images = None
        self.ages = None
        self.train = train
        if entire:
            self.datalen = 66891
            self.index_start = 0
        else:
            if train:
                self.datalen = 50000
                self.index_start = 0
            else:
                self.datalen = 16891
                self.index_start = 50000
        self.chunk_size = 5000
        self.data_pre_path = data_pre_path
        self.transform = transform

    def __isIndexOpened(self, index):
        if self.chunk_opened is None:
            return False
        return ((index >= self.chunk_opened * self.chunk_size) and (index < (self.chunk_opened+1) * self.chunk_size))
        
    def __indexToChunk(self, index):
        return index//self.chunk_size
    
    def __load_chunk(self, i):
        i = int(i)
        if i == self.chunk_opened:
            return
        else:
            file = h5py.File(self.data_pre_path + str(i) + '.h', "r+")
            self.images = np.array(file["/images"]).astype("uint8")
            self.ages = np.array(file["/meta"]).astype("uint8")
            self.chunk_opened = i

    def visualize(self, i):
        i += self.index_start
        im, age = self.__getitem__(i)
        im = im.detach().cpu().numpy()
        plt.figure()
        plt.title(str(age))
        plt.imshow(np.swapaxes(im, 0, 2))
        plt.show()

    def __getitem__(self, i):
        i += self.index_start
        if not self.__isIndexOpened(i):
            self.__load_chunk(self.__indexToChunk(i))
        return self.transform(self.images[i%self.chunk_size]), torch.DoubleTensor([self.ages[i%self.chunk_size]])

    def __len__(self):
        return self.datalen


class IMDB_Face_Both(Dataset):
    """docstring for IMDB_Face"""
    def __init__(self, old_data_pre_path = None, young_data_pre_path = None, transform=lambda x:x, train = True, entire=False):
        super(IMDB_Face_Both, self).__init__()
        if old_data_pre_path is None:
            self.old_set = IMDB_Face_Old(transform = transform, train=train, entire=entire)
        else:
            self.old_set = IMDB_Face_Old(data_pre_path = old_data_pre_path,transform = transform, train=train, entire=entire)
        
        if young_data_pre_path is None:
            self.young_set = IMDB_Face_Young(transform = transform, train=train, entire=entire)
        else:
            self.young_set = IMDB_Face_Young(data_pre_path = young_data_pre_path,transform = transform, train=train, entire=entire)

        self.oldlen = len(self.old_set)
        self.younglen = len(self.young_set)

        self.datalen = self.oldlen + self.younglen

        # one for old, zero for old
        self.choice = np.zeros((self.datalen,))
        self.choice[:self.oldlen] = 1
        np.random.shuffle(self.choice)
        self.choicetoindex = self.choice.copy()
        younindices = self.choice == 0
        oldindices = self.choice == 1
        self.choicetoindex[younindices] = np.arange(self.younglen).astype(np.int)
        self.choicetoindex[oldindices] = np.arange(self.oldlen).astype(np.int)

    def visualize(self, i):
        index = int(self.choicetoindex[i])
        if self.choice[i] == 0:
            im, age = self.young_set[index]
        else:
            im, age = self.old_set[index]
        im = im.detach().cpu().numpy()
        plt.figure()
        plt.title(str(age))
        plt.imshow(np.swapaxes(im, 0, 2))
        plt.show()

    def __getitem__(self, i):
        index = int(self.choicetoindex[i])
        if self.choice[i] == 0:
            im,age = self.young_set[index]
        else:
            im,age = self.old_set[index]
        return im, age, int(self.choice[i])

    def __len__(self):
        return self.datalen

