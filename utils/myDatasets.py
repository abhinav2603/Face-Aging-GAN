import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import pickle
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import PIL

class IMDB_Face(Dataset):
    """docstring for IMDB_Face"""
    def __init__(self, data_path = '../datasets/10-25.pickle', transform=lambda x:x):
        super(IMDB_Face, self).__init__()
        with open(data_path, 'rb') as handle:
            dic = pickle.load(handle)
            self.ages = dic['ages']
            self.image_paths = dic['image_paths']
            self.length = dic['len']
          
        self.transform = transform
        
    def __getitem__(self, i):
        image = plt.imread(self.image_paths[i])
        return image, self.ages[i]

    def __len__(self):
        return self.length

    def visualize(self, i):
        image = plt.imread(self.image_paths[i])
        plt.figure()
        plt.imshow(image)
        plt.title('Age = {}'.format(self.ages[i]))
        plt.show()


