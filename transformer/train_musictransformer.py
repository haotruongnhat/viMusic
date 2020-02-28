import os, sys
import numpy as np 
import torch 
import argparse

from MTransformer.music_transformer.dataloader import MusicDataBunch, Midi2ItemProcessor, batch_position_tfm

from MTransformer.numpy_encode import *
from MTransformer import config
from MTransformer import music_transformer
# from musicautobot.music_transformer import *
from MTransformer.utils.midifile import *
from MTransformer.utils.file_processing import process_all

import fastai 
from fastai.callbacks import SaveModelCallback
# torch.cuda.set_device(1)

midi_path = '../data/midi'
data_path = '../data/npy'
data_save_name= 'npy_data.pkl'




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do data processing and training Music Transformer from midi files')
    parser.add_argument('--input_dir', default='../data/npy', help='path to folder contain .npy data file')
    parser.add_argument('--data_savename', default='classical_data_save.pkl', help='name to save pkl file')
    parser.add_argument('--model_savepath', default='../data/npy', help='path to save pkl file')
    parser.add_argument('--pretrained_path', default=None, help='path to pretrain model')
    parser.add_argument('--mode', default='train', help='choose between train/predict mode, default: train')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--config', default='default', help='config')
    parser.add_argument('--lr', default = 1e-02, type=float, help='learning rate')
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch for training')
    parser.add_argument('--encode_position', default=True, type=bool)
    args = parser.parse_args()

    if os.path.isdir(args.model_savepath):
        os.makedirs(args.model_savepath)

    
    # midi_files = fastai.data_block.get_files(midi_path, '.mid', recurse=True)
    processors = [Midi2ItemProcessor()]
    
    # Process for training step
    callback = ['accuracy']
    batch_size = args.batch_size
    encode_position = args.encode_position
    lr = args.lr
    dl_tfms = [batch_position_tfm] if encode_position else []
    data = music_transformer.load_data(args.input_dir, args.data_savename, bs=batch_size, 
                                        encode_position=encode_position, dl_tfms=dl_tfms)

    
    cfg = config.default_config()
    
    cfg['encode_position'] = encode_position
    learn = music_transformer.music_model_learner(data, pretrained_path=args.pretrained_path, config=cfg)
    # learn.lr_find(start_lr=1e-07, end_lr= 0.1, num_it=100)
    
    # learn.fit(lr=lr, epochs= args.epoch, callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])
    learn.fit_one_cycle(max_lr= 0.0001, cyc_len = args.epoch, callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')] )
