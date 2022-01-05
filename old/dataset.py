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
from audio2numpy import open_audio
import librosa

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def absoluteFilePaths(directory):
    l = []
    for filenames in os.listdir(directory):
        l.append(os.path.abspath(os.path.join(directory, filenames)))
    return l
        
            
            
class shift_by_random(object):
    '''shifts by random number'''
    def __init__(self, height):
        self.height = height
        
    def __call__(self, sample):
        data, label = sample
        to_shift = np.random.choice(range(self.height))
        data = torch.roll(data, int(to_shift), dims = 1)
        return data,label


    
    
    
##### New dataset
# randomly split the audio files into train and validation per raga
# open the audio randomly get 30 second chunk
# randomly change pitch by up to half octave either way (training only)
# extract harmonic 
# cqt transform
# convert to float 
# normailize 
    
class Ragam_Dataset(torch.utils.data.Dataset):
   
    def __init__(self,path = "/gpfs/data1/cmongp1/mpaliyam/raganet/data/audio",
                 train = True, transform = None, test_size = .15):
        print('Initializing dataset:')  
        self.transform = transform
        train_songs = []
        train_labels = []
        val_songs = []
        val_labels = []
        ragam_names = {}
        self.train = train
        
        for ragam_id,p in enumerate(listdir_nohidden(path)):
            ragam_names[ragam_id] = p
            songs = absoluteFilePaths(os.path.join(path, p))
            for s in songs: 
                if not s.endswith('.mp3'):
                    songs.remove(s)
                    print(s, ' removed.')
                    
            ind = int(len(songs) * (1 - test_size))
            l = np.tile(ragam_id, len(songs))
            
            train_songs.extend(songs[:ind])
            train_labels.extend(l[:ind])
            val_songs.extend(songs[ind:])
            val_labels.extend(l[ind:])
            
        if train:
            self.data = train_songs
            self.labels = np.array(train_labels).astype(float)
            print("Train set has ", len(self.labels), "labels.")
            unique, counts = np.unique(self.labels, return_counts=True)
            print(dict(zip(unique, counts)))
        else:
            self.data = val_songs
            self.labels = np.array(val_labels).astype(float)
            print("Validation set has ", len(self.labels), "labels.")
            unique, counts = np.unique(self.labels, return_counts=True)
            print(dict(zip(unique, counts)))
            
        self.ragam_names = ragam_names
        print('Finished initializing: ', self.ragam_names)        
            
    def __len__(self):
        return len(self.labels)            
            
        
    def __getitem__(self, idx): 
        hop_length = 1024
        # open audio
        file_path = self.data[idx]
        signal, sampling_rate = open_audio(file_path)
        if len(signal.shape) > 1: 
            signal = np.mean(signal, axis = 1)
        if sampling_rate != 44100:
            signal = librosa.resample(signal, sampling_rate, 44100)
            sampling_rate = 44100
            
            
        # get 30 second chunk
        len_index_30_sec = int(30 / (1 / sampling_rate))
        # trim first and last 30 seconds 
        signal = signal[len_index_30_sec:-len_index_30_sec]
        # random start index
        start_index = np.random.randint(low = 0, high = len(signal) - len_index_30_sec)
        signal = signal[start_index:start_index + len_index_30_sec]
        # if training change pitch randomly
        if self.train:
            n_steps = np.random.randint(low = -4, high=4) 
            signal = librosa.effects.pitch_shift(signal, sampling_rate, n_steps=n_steps)
        # extract harmonic 
        data_h = librosa.effects.harmonic(signal)
        # cqt transform
        S = np.real(librosa.cqt(data_h, sr=sampling_rate, hop_length=hop_length)).astype(np.float32)

        
        d = torch.from_numpy(np.expand_dims(S, axis = 0)).type(torch.FloatTensor) 
        # normalize 
        d = F.normalize(d)
        l = torch.from_numpy(np.array(self.labels[idx])).type(torch.LongTensor)
#         print(d.shape, sampling_rate, file_path)

        return d,l
    
    
    
    
    

class Ragam_Dataset_old(torch.utils.data.Dataset):
   
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
                
                
            
            
            
