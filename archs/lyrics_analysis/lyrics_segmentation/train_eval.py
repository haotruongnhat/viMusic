from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch

import numpy as np
import os
import argparse

from pipelines.data_processing.configs import pop_lyrics_analysis_config, rock_lyrics_analysis_config
from pipelines.data_processing.dataset import LyricsDataset

from archs.lyrics_analysis.lyrics_segmentation.utils.model import LyricsSegmentationModel
from archs.lyrics_analysis.lyrics_segmentation.utils.dataloader import LyricsDataLoader

from common.constants import COMPOSER

import argparse

parser = argparse.ArgumentParser(description='Trainer for lyrics segmentation')
parser.add_argument('--genre',type=str,default='pop')
parser.add_argument('--checkpoint_path',type=str,default='')
parser.add_argument('--save_path',type=str,default='./models/lyrics_analysis/checkpoints/')
parser.add_argument('--summary_path',type=str,default='./archs/lyrics_analysis/lyrics_segmentation/utils/log')

def train():   
    ### Parse argument
    args = parser.parse_args()
    ###0. Define configuration according to genre
    genre_config = {
        COMPOSER.POP : pop_lyrics_analysis_config,
        COMPOSER.ROCK : rock_lyrics_analysis_config
    }

    ###1. Define the model
    model = LyricsSegmentationModel(args.genre,config=genre_config[args.genre]) #model
    optimizer = Adam(model.parameters(),lr=genre_config[args.genre].hparams.learning_rate) #opt
    loss = BCELoss()

    ###2. Create dataset and dataloader
    dataset = LyricsDataset(genre_config[args.genre],genre_config[args.genre].dataset_dir)
    train_dataset,test_dataset = dataset.split_dataset()
    train_dataloader = LyricsDataLoader(train_dataset,genre_config[args.genre])
    test_dataloader = LyricsDataLoader(test_dataset,genre_config[args.genre])
    ###3. Checking GPU
    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()

    ###4. Check if log is enabled
    if args.summary_path != '':
        writer = SummaryWriter(args.summary_path)

    ###5. Begin training
    model.train()
    for epoch in range(genre_config[args.genre].hparams.epoch):
        #reset gradient of the optimizer
        optimizer.zero_grad()
        running_loss = 0
        #Training loop in each batch
        for i_batch,(batch_input,batch_label) in enumerate(train_dataloader):
            #Putting them on GPU
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
                batch_label = batch_label.cuda()

            #squeeze
            batch_input = batch_input.view(batch_input.shape[0] * batch_input.shape[1],batch_input.shape[2],batch_input.shape[3]).float()
            batch_label = batch_label.view(batch_label.shape[0] * batch_label.shape[1],batch_label.shape[2]).float()
            
            #forwarding
            batch_output = model(batch_input)

            #calculate loss
            criterion=loss(batch_output,batch_label)

            #Get metrics
            accuracy = LyricsSegmentationModel.accuracy(batch_output,batch_label)
            precision = LyricsSegmentationModel.precision(batch_output,batch_label)
            recall = LyricsSegmentationModel.recall(batch_output,batch_label)
            f1 = LyricsSegmentationModel.f1(batch_output,batch_label)
            # print("Epoch {} | Batch {} | Accuracy {} | Precision {} | Recall {} | F1 {}".format(epoch + 1,i_batch + 1,accuracy,precision,recall,f1))

            #accumulating BCE loss
            running_loss += criterion.item()

            #backward and step to update weight
            criterion.backward()
            optimizer.step()
        #End of an epoch, update loss
        running_loss /= len(train_dataloader)
        print('Epoch {} | Train Loss {} |'.format(epoch + 1,running_loss + 1))
        #Update metric
        writer.add_scalar('Train loss',running_loss,epoch)

    model.eval()
    running_loss = 0
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    #Save model
    torch.save(model.state_dict(), os.path.join(args.save_path,args.genre,datetime.now().strftime("%d_%m_%Y_%H-%M-%S") + ".pt"))

    for i_batch,(batch_input,batch_label) in enumerate(test_dataloader):
        #GPU
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_label = batch_label.cuda()
        
        #squeeze
        batch_input = batch_input.view(batch_input.shape[0] * batch_input.shape[1],batch_input.shape[2],batch_input.shape[3]).float()
        batch_label = batch_label.view(batch_label.shape[0] * batch_label.shape[1],batch_label.shape[2]).float()

        #forwarding
        batch_output = model(batch_input)

        #calculate loss
        criterion=loss(batch_output,batch_label)

        accuracy = LyricsSegmentationModel.accuracy(batch_output,batch_label)
        precision = LyricsSegmentationModel.precision(batch_output,batch_label)
        recall = LyricsSegmentationModel.recall(batch_output,batch_label)
        f1 = LyricsSegmentationModel.f1(batch_output,batch_label)

        running_loss += criterion.item()

        print("Evaluation: Loss {} | Accuracy {} | Precision {} | Recall {} | F1 {}".format(running_loss,accuracy,precision,recall,f1))
    writer.close()


if __name__=='__main__': 
    train()