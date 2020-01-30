from musicautobot.numpy_encode import *
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.utils.midifile import *
from musicautobot.utils.file_processing import process_all
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

train_data_path = '../../data/cached' # path to folder contain pickle pkl data files
data_save_name = 'Rock_data_save.pkl' # pickle data that contain all enconded midi file
config = default_config() # get config 
num_epoch = 10
batch_size = 16

def train(num_epoch, train_data_path, file_name, config, pretrained_path:Str=None):
    encode_position = True
    dl_tfms = [batch_position_tfm] if encode_position else []
    data = load_data(data_path, data_save_name, bs=batch_size, encode_position=encode_position, dl_tfms=dl_tfms)
    config['encode_position'] = encode_position
    print(config)
    if pretrain_path: 
        learn = music_model_learner(data, config=config.copy(), pretrained_path)
    learn.fit_one_cycle(num_epoch)
    
if __name__='__main__':
    train(num_epoch,train_data_path, file_name, config)