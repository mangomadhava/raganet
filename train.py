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
    parser.add_argument('--bs', type = int, help = 'batch size')
    parser.add_argument('--epochs', type = int, help = 'epochs')
    parser.add_argument('--stage', help = 'which stage, cnn, lstm, or both')
    parser.add_argument('--name', help = 'Name of experiment')
    options = parser.parse_args()

    # set random seed for repeatable trials
    torch.manual_seed(8)
    
    # run on gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('--------------- MODEL ---------------')
    if options.stage == 'both':
        model = ragam_identifier_extractor_cnn_lstm(num_ragas = 2)
    elif options.stage == 'cnn':
        model = extractor_cnn(dropout = .1, seperate = True)
    elif options.stage == 'raganet':
        model = raganet(num_ragas = 2)
        
    if torch.cuda.device_count() > 1: 
        model = nn.DataParallel(model)
        model.to(f'cuda:{model.device_ids[0]}')
    else: 
        model = model.to(device)
        
    print(summary(model, (1,12,600)))
    print("Running on ", device)
          
    
    optim = torch.optim.Adam(model.parameters(), lr=options.lr)
#     optim = torch.optim.SGD(model.parameters(), lr = options.lr, momentum = .01)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[], gamma=.1)
    print('Optimizer: ', optim)
    
    epochs = options.epochs
    print('Epochs: ', epochs)
    
    loss_fn = nn.CrossEntropyLoss()
    print('Loss Function: ', loss_fn)
    
    batch_size = options.bs
    print('Batch Size: ', batch_size)
    
    

    print('--------------- DATA ---------------')
    train_data = Ragam_Dataset(train = True,
                               path = '/gpfs/data1/cmongp1/mpaliyam/raganet/data/numpy_files')
    
    val_data = Ragam_Dataset(train = False, 
                             path = '/gpfs/data1/cmongp1/mpaliyam/raganet/data/numpy_files')
    
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4)
    
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = 2)
    
    '''training/validation loop'''
    print('--------------- TRAINING ---------------')
    writer = SummaryWriter('./runs/' + str(options.name) + '_LR_' + str(options.lr)
           + '_BS_' + str(options.bs)+ '_epochs_' + str(options.epochs) + '_stage_' + str(options.stage))
    

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = []
        for batch in train_loader:
            data, label = batch
            data = data.to(device)
            label = label.to(device)
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
        correct = torch.tensor([0])
        for batch in val_loader:
            with torch.no_grad():
                data, label = batch
                data = data.to(device)
                label = label.to(device) 
                model_output = model(data)
                if options.stage == 'cnn':
                    label = label.repeat_interleave(10)
                loss = loss_fn(model_output, label)
                model_pred = torch.argmax(model_output, dim = 1)
                correct += (model_pred == label).sum()
                
                val_loss.append(loss.item())
                
        acc = (correct.item() / len(val_data)) * 100
        writer.add_scalar('Loss/train', np.mean(train_loss), epoch)
        writer.add_scalar('Loss/validation', np.mean(val_loss), epoch)
        writer.add_scalar('Accuracy/validation', acc, epoch)
        
#         scheduler.step()
    

    torch.save(model.state_dict(), './trained_model' + options.name + '.tar')
    
    
    
