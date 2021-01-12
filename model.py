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
from collections import OrderedDict



class m_block(nn.Module):
    def __init__(self, inplanes, planes, y_k_s = 5):
        super(m_block, self).__init__()
        p_y = y_k_s // 2
        self.conv_s1 = nn.Conv2d(inplanes, planes, kernel_size=(1,y_k_s), padding=(0,p_y), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_s2 = nn.Conv2d(inplanes, planes, kernel_size=(3,y_k_s), padding=(1,p_y), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv_s3 = nn.Conv2d(inplanes, planes, kernel_size=(5,y_k_s), padding=(2,p_y), bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.project = nn.Sequential(
            nn.Conv2d(3 * planes, planes, 1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(planes, planes, kernel_size=(1,3), padding=(0,1), bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        d_1_feat = self.conv_s1(x)
        d_2_feat = self.conv_s2(x)
        d_3_feat = self.conv_s3(x)
        feat_vect = torch.cat([d_1_feat, d_2_feat, d_3_feat], dim = 1)
        # residual connection
        out = self.project(feat_vect) + x 
        return out
        
class raganet(nn.Module):
    def __init__(self, height = 12, num_ragas = 2):
        super(raganet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,3), padding=(0,1), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2,4), padding=1)
        
        self.b1 = m_block(32, 64, 5)
        self.b2 = m_block(64, 64, 3)
        self.b3 = m_block(64, 128, 3)
        self.fc = nn.Linear(2560, num_ragas)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
#         print(x.shape)
        x = self.b1(x)
        x = self.maxpool(x)
#         print(x.shape)
        x = self.b2(x)
        x = self.maxpool(x)
#         print(x.shape)
        x = self.b3(x)
        x = self.maxpool(x)
#         print(x.shape)
        x = x.view(x.size(0), -1)
#         print(x.shape)
        x = self.fc(x)
        return x

class feature_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout,num_ragas, num_layers=2):
        super(feature_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,batch_first=True)
        self.dropout = nn.Dropout(p = dropout)
        self.hidden = self.init_hidden()
        self.linear = nn.Linear(hidden_dim, num_ragas)
        
    def init_hidden(self):
        return (torch.zeros(self.num_layers, 1, self.hidden_dim),
                torch.zeros(self.num_layers, 1, self.hidden_dim))
    
    def forward(self, x):
        x = self.dropout(x)
        lstm_out, self.hidden = self.lstm(x)
        x = self.linear(lstm_out[:,-1,:])
        return x



class extractor_cnn(nn.Module):
    def __init__(self, dropout = 0.1, height = 24, seperate = False, num_ragas = 2):
        super(extractor_cnn, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('norm0', nn.BatchNorm2d(1)),
            
            ('conv1', nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3)),
            ('norm1', nn.BatchNorm2d(4)),
            ('relu1', nn.LeakyReLU()),
            ('drop1', nn.Dropout(p=dropout)),

                        
            ('conv2', nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3)),
            ('norm2', nn.BatchNorm2d(4)),
            ('relu2', nn.LeakyReLU()),
            ('drop2', nn.Dropout(p=dropout)),

               
            ('conv3', nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3)),
            ('norm3', nn.BatchNorm2d(4)),
            ('relu3', nn.LeakyReLU()),
            ('drop3', nn.Dropout(p=dropout)),

            
            ('adaptivepool2', nn.AdaptiveMaxPool2d((9,9))),
            
            ('conv4', nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3)),
            ('norm4', nn.BatchNorm2d(8)),
            ('relu4', nn.LeakyReLU()),
            ('drop4', nn.Dropout(p=dropout)),

            
            ('conv5', nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3)),
            ('norm5', nn.BatchNorm2d(8)),
            ('relu5', nn.LeakyReLU()),
            
            ('pool2', nn.AvgPool2d(kernel_size = 3)),
        ]))
        
        self.last_layer = False
        if seperate:
            self.last_layer = True
            self.fc = nn.Linear(8,num_ragas)
          
            
            
                
    def forward(self, x):
        if self.last_layer:
            batch_size = x.shape[0]
            seq_length = x.shape[1]
            height = x.shape[2]
            width = x.shape[3]
            x = x.reshape(batch_size * seq_length, height, width)
            x = x.unsqueeze(1)
            
        x = self.encoder(x)
        
        
        if self.last_layer:
            x = x.flatten(1)
            x = self.fc(x)
            
        return x

class ragam_identifier_extractor_cnn_lstm(nn.Module):
    def __init__(self, num_ragas, dropout = .1, height = 24, input_dim_lstm = 64, hidden_dim_lstm = 32):
        super(ragam_identifier_extractor_cnn_lstm, self).__init__()
        
        self.chunk_encoder = extractor_cnn(dropout = dropout, height = height)
        self.raga_lstm = feature_LSTM(input_dim = input_dim_lstm, dropout = dropout, num_ragas = num_ragas, hidden_dim = hidden_dim_lstm)
        
        
    '''x has shape batchsize,seq_length,height,width'''
    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        x = x.reshape(batch_size * seq_length, height, width)
        x = x.unsqueeze(1)
        x = self.chunk_encoder(x)
        x = x.reshape(batch_size, seq_length, -1)
        x = self.raga_lstm(x)
        return x





class music_motivated_cnn(nn.Module):
    def __init__(self, dropout = .1, num_ragas = 2):
        super(music_motivated_cnn, self).__init__()
        
        self.norm0 = nn.BatchNorm2d(1)
        
        self.temp_conv_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1,16))
        self.temp_conv_2 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1,32))
        self.temp_conv_3 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1,64))

        self.freq_conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (32,3))
        self.freq_conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (64,3))


        self.avgpool = nn.AdaptiveAvgPool2d(())
