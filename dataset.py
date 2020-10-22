import torch
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from data import *
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
   
    def __init__(self,path = "/Users/madhavapaliyam/Documents/CMSC/rando/raganet/data/pickle_files", transpose = False, length = 1500, chunk_size = 150, train = True,transform = None):
        
        print('Initializing dataset:')

        self.transform = transform
        
        data = []
        labels = []
        ragam_names = []
        
        for ragam_id, p in enumerate(os.listdir(path)):
            if p[0] != '.':
                ragam_names.append(p)
                file = open(os.path.join(path,p), "rb")
                ragam_info = pickle.load(file)
                for i,song_sttft in enumerate(ragam_info):
                    for x in range(0, song_sttft.shape[1], length):
                        slice = song_sttft[:,x : x + length]
                        if slice.shape[1] == length:
                            song_chunks = []
                            for j in range(0, slice.shape[1], chunk_size):
                                chunk = slice[:,j : j + chunk_size]
                                if chunk.shape[1] == chunk_size:
                                    song_chunks.append(chunk)
                                    
                            data.append(np.array(song_chunks))
                            labels.append(ragam_id - 1)
                print('Ragam pickle procesed: ', p)
        
        
        self.data = torch.from_numpy(np.array(data))
        self.labels = torch.from_numpy(np.array(labels))
        
        X_train, X_val, y_train, y_val = train_test_split(self.data, self.labels, test_size=0.15, random_state=4)
        
        if train:
            self.data = X_train
            self.labels = y_train
        else:
            self.data = X_val
            self.labels = y_val
        print(len(self.labels), ' chunks.')
        self.ragam_names = ragam_names
        print('Finished initializing: ', self.ragam_names)
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        if self.transform:
            return self.transform((self.data[idx],self.labels[idx]))
            
        return self.data[idx], self.labels[idx]
                
                
            
            
            
