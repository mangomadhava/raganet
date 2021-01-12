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
                 train = True, transform = None):
        
        print('Initializing dataset:')

        self.transform = transform
        
        data = []
        labels = []
        ragam_names = {}
        
        for ragam_id, p in enumerate(os.listdir(path)):
            if p[0] != '.':
                ragam_names[ragam_id] = p[:-3]
                npy_file_data = np.load(os.path.join(path, p))
                l = np.tile(ragam_id, npy_file_data.shape[0])
                labels.extend(l)
                data.extend(npy_file_data)
                
        self.data = np.array(data).astype(float)
        self.labels = np.array(labels).astype(float)
        
        X_train, X_val, y_train, y_val = train_test_split(self.data, self.labels, 
                                        test_size=0.15, random_state=4, shuffle = True)
        
        if train:
            self.data = X_train
            self.labels = y_train
            print("Train set has ", len(self.labels), "labels.")
            unique, counts = np.unique(self.labels, return_counts=True)
            print(dict(zip(unique, counts)))
        else:
            self.data = X_val
            self.labels = y_val
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
            
        return self.data[idx], self.labels[idx]
                
                
            
            
            
