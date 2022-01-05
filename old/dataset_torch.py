import torch
import sys
sys.path.append('../')
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import IPython.display as ipd
import numpy as np
import os
torchaudio.set_audio_backend("sox_io")


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def absoluteFilePaths(directory):
    l = []
    for filenames in os.listdir(directory):
        l.append(os.path.abspath(os.path.join(directory, filenames)))
    return l



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
        self.sample_rate = 44100
        self.mel_model = torch.nn.Sequential(
                T.MelSpectrogram(sample_rate = self.sample_rate,n_mels = 24))
        
    def __len__(self):
        return len(self.labels)            
            
        
    def __getitem__(self, idx): 
        waveform, sample_rate = torchaudio.load(self.data[idx])
        if len(waveform.shape) > 1: 
            waveform = waveform.mean(axis = 0).reshape((1,-1))
   
        # get 30 second chunk
        len_index_30_sec = int(30 / (1 / sample_rate))
        # trim first and last 30 seconds if long enough
        if waveform.shape[1] > 2 * len_index_30_sec:
            waveform = waveform[:, len_index_30_sec:-len_index_30_sec]
            # get random start index
            start_index = np.random.randint(low = 0, high = waveform.shape[1] - len_index_30_sec)
            waveform = waveform[:, start_index:start_index + 2*(len_index_30_sec)]
        else: 
            waveform = waveform[:,0: 2 * len_index_30_sec]
            print('too short', waveform.shape, sample_rate)
        if self.train:
            # pitch shift and change sampling rate to 44.1
            effects = [
                ["pitch", "-q", "300"],
                [ "rate", "44100"]
                ]
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, effects)
        else: 
            # only change sampling rate to 44.1
            effects = [
                [ "rate", "44100"]
                ]
            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, effects)
            
        assert sample_rate == self.sample_rate
        # recalculate 30 second length bc of new sample rates
        len_index_30_sec = int(30 / (1 / sample_rate))
        waveform = waveform[:, 0:len_index_30_sec]
#         print(sample_rate, waveform.shape)
        label = torch.from_numpy(np.array(self.labels[idx])).type(torch.LongTensor)
        return waveform, label
        
        
        