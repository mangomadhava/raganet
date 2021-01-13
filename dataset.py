import torch
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import RandomSampler
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split



class shift_by_random(object):
    '''shifts by random number'''
    def __init__(self, height):
        self.height = height
        
    def __call__(self, sample):
        data, label = sample
        to_shift = np.random.choice(range(self.height))
        data = torch.roll(data, int(to_shift), dims = 1)
        return data,label



class Ragam_Dataset(torch.utils.data.Dataset):
   
    def __init__(self,path = "/Users/madhavapaliyam/Documents/CMSC/rando/raganet/data/npy",
                 train = True, transform = None, test_size = .15):
        
        print('Initializing dataset:')

        self.transform = transform
        
        data_train = []
        labels_train = []
        data_val = []
        labels_val = []
        ragam_names = {}
        
        for ragam_id, p in enumerate(os.listdir(path)):
            if p[0] != '.':
                ragam_names[ragam_id] = p[:-3]
                npy_file_data = np.load(os.path.join(path, p))
                l = np.tile(ragam_id, npy_file_data.shape[0])
                index = int(npy_file_data.shape[0] - (npy_file_data.shape[0] * test_size))
                data_train.extend(npy_file_data[:index])
                data_val.extend(npy_file_data[index:])
                labels_train.extend(l[:index])
                labels_val.extend(l[index:])
                
#         self.data = np.array(data).astype(float)
#         self.labels = np.array(labels).astype(float)
        
#         X_train, X_val, y_train, y_val = train_test_split(self.data, self.labels, 
#                                         test_size=0.15, random_state=4, shuffle = False)
        
        
        if train:
            self.data = np.array(data_train).astype(float)
            self.labels = np.array(labels_train).astype(float)
            print("Train set has ", len(self.labels), "labels.")
            unique, counts = np.unique(self.labels, return_counts=True)
            print(dict(zip(unique, counts)))
        else:
            self.data = np.array(data_val).astype(float)
            self.labels = np.array(labels_val).astype(float)
            print("Validation set has ", len(self.labels), "labels.")
            unique, counts = np.unique(self.labels, return_counts=True)
            print(dict(zip(unique, counts)))
            
        self.ragam_names = ragam_names
        print('Finished initializing: ', self.ragam_names)
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        if self.transform:
            return self.transform((self.data[idx],self.labels[idx]))
        d = torch.from_numpy(np.expand_dims(self.data[idx], axis = 0)).type(torch.FloatTensor) 
        l = torch.from_numpy(np.array(self.labels[idx])).type(torch.LongTensor)
        return d,l
                
                
            
            
            
