import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, ReLU, Linear, Tanh, Softmax, Sigmoid, Dropout, LSTM

import numpy as np
import math

from pipelines.data_processing.configs import pop_lyrics_analysis_config


class LyricsSegmentationModel(torch.nn.Module):
    #This is a dumb model using CNN to predict segments
    def __init__(self,genre,config=pop_lyrics_analysis_config,mode='train'):
        super(LyricsSegmentationModel,self).__init__()

        #Defning parameters
        self._channels = 1 #default
        self._channels_out = 1

        self._input_height = 2 * config.w_size
        self._input_width = config.max_ssm_size

        self._conv_1_kernel_width = config.w_size + 1
        self._conv_1_kernel_height = config.w_size + 1
        self._max_pool_1_kernel_width = config.w_size
        self._max_pool_1_kernel_height = config.w_size

        self._conv_2_kernel_width = config.w_size
        self._conv_2_kernel_height = 1
        self._max_pool_2_kernel_width = 5 #according to (2*w_size,max_ssm_size)
        self._max_pool_2_kernel_height = 5

        self._output_size = config.output_size
        

        ###A. Define CNN layer
        ###Be careful for input size being too small
        self._conv_1 = Conv2d(
            in_channels=self._channels,
            out_channels=self._channels_out,
            kernel_size=(self._conv_1_kernel_height,self._conv_1_kernel_width),
            stride=3)
        self._relu_1 = ReLU()
        self._max_1 = MaxPool2d(
            kernel_size=(self._max_pool_1_kernel_height,self._max_pool_1_kernel_width),
            padding=1,
            stride=3)

        self._conv_2 = Conv2d(
            in_channels=self._channels_out,
            out_channels=self._channels,
            kernel_size=(self._conv_2_kernel_height,self._conv_2_kernel_width),
            stride=4)
        self._relu_2 = ReLU()
        self._max_2 = MaxPool2d(
            kernel_size=(self._max_pool_2_kernel_height,self._max_pool_2_kernel_width),padding=2,
            stride=4)

        self._dropout_1 = Dropout(config.hparams.dropout)

        ###B. Defnining linear layer
        self._linear_hidden_layer = 32
        self._linear_layers = torch.nn.Sequential(
            Linear(1,self._linear_hidden_layer),
            ReLU(),
            Linear(self._linear_hidden_layer,self._linear_hidden_layer),
            ReLU(),
            Linear(self._linear_hidden_layer,2),
            Softmax(dim=0) #calculate on "batch_size" dim
        )

        self._dropout_2 = Dropout(config.hparams.dropout)

        # ###B. Defning LSTM layer
        # self._lstm = LSTM(1,1,512) #input = 1, output = 1, layer = 512
        # ###Defining h0 and c0 to keep initial state throughout each batch
        # self._h0 = torch.rand(512, 1, 1)
        # self._c0 = torch.rand(512, 1, 1)
        # if torch.cuda.is_available():
        #     self._h0 = self._h0.cuda()
        #     self._c0 = self._c0.cuda()
        self._mode = mode

    def forward(self,x):


        #Padding in case the input tensor is too small
        #x.shape == [batch_size,2 * w_size,input_size]
        #unsqueeze to add extra one channel
        x = x.unsqueeze(1).float()

        #cnn_layer
        x = self._conv_1(x)
        x = self._relu_1(x)
        x = self._max_1(x)
        x = self._conv_2(x)
        x = self._relu_2(x)
        x = self._max_2(x)

        #dropout
        if self._mode == 'train':
            x = self._dropout_1(x)
        #view again
        x = x.view(x.shape[0],-1)
        
        #going through linear
        x = self._linear_layers(x)
        #x.shape == [1,batch_size]
        if self._mode == 'train':
            x = self._dropout_2(x)

        return x

    @staticmethod
    def accuracy(predicted,logit):
        
        #shape == [batch_size,2]
        num_element = predicted.shape[0]
        output_predicted = torch.argmax(predicted,dim=1)
        output_logit = torch.argmax(logit,dim=1)
        return sum(output_predicted == output_logit).float() / num_element

    @staticmethod
    def precision(predicted,logit):
        #shape == [batch_size,2]
        output_predicted = torch.argmax(predicted,dim=1)
        output_logit = torch.argmax(logit,dim=1)
        true_pos = ((output_predicted + output_logit) > 1.).sum().float()
        false_pos = ((output_predicted - output_logit) > 0.).sum().float()
        return true_pos / (true_pos + false_pos)

    @staticmethod
    def recall(predicted,logit):
        #shape == [batch_size,2]
        output_predicted = torch.argmax(predicted,dim=1)
        output_logit = torch.argmax(logit,dim=1)
        true_pos = ((output_predicted + output_logit) > 1.).sum().float()
        false_neg = ((output_logit - output_predicted) > 0.).sum().float()
        return true_pos / (true_pos + false_neg)

    @staticmethod
    def f1(predicted,logit):
        precision = LyricsSegmentationModel.precision(predicted,logit)
        recall = LyricsSegmentationModel.recall(predicted,logit)
        return 2 * (precision * recall) / (precision + recall)


