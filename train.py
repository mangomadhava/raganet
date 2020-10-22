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
from torchvision import transforms, utils


from dataset import *
from model import *




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--lr', type = float, help = 'Learning rate')
    parser.add_argument('--bs', type = int, help = 'Learning rate')
    parser.add_argument('--epochs', type = int, help = 'Learning rate')
    parser.add_argument('--stage', help = 'which stage, cnn, lstm, or both')
    parser.add_argument('--name', help = 'Name of expiriment')
    options = parser.parse_args()

    
    print('--------------- MODEL ---------------')
    if options.stage == 'both':
        model = ragam_identifier_extractor_cnn_lstm(num_ragas = 2)
    elif options.stage == 'cnn':
        model = extractor_cnn(dropout = .1, seperate = True)
    print(model)
    
#    optim = torch.optim.Adam(model.parameters(), lr=options.lr)
    optim = torch.optim.SGD(model.parameters(), lr = options.lr, momentum = .01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[15], gamma=.1)
    print('Optimizer: ', optim)
    
    epochs = options.epochs
    print('Epochs: ', epochs)
    
    loss_fn = nn.CrossEntropyLoss()
    print('Loss Function: ', loss_fn)
    
    batch_size = options.bs
    print('Batch Size: ', batch_size)
    
    

    print('--------------- DATA ---------------')
    train_data = Ragam_Dataset(train = True, transform = transforms.Compose([shift_by_random(24)]))
    
    val_data = Ragam_Dataset(train = False)
    
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 3)
    
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers = 2)
    
    '''training/validation loop'''
    print('--------------- TRAINING ---------------')
    print('Logging to: ', options.name)
    writer = SummaryWriter('./runs/' + options.name)
    

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = []
        for batch in train_loader:
            data, label = batch
            model_output = model(data)
            if options.stage == 'cnn':
                label = label.repeat_interleave(10)
            loss = loss_fn(model_output, label)
            train_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            
        model.eval()
        val_loss = []
        for batch in val_loader:
            with torch.no_grad():
                data, label = batch
                model_output = model(data)
                if options.stage == 'cnn':
                    label = label.repeat_interleave(10)
                loss = loss_fn(model_output, label)
                val_loss.append(loss.item())
            
        writer.add_scalar('Loss/train', np.mean(train_loss), epoch)
        writer.add_scalar('Loss/validation', np.mean(val_loss), epoch)
        
        scheduler.step()
    

    torch.save(model.state_dict(), './trained_model' + options.name + '.tar')
    
    
    
