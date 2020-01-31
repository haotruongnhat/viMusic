import os 

from musicautobot.numpy_encode import *
from musicautobot.config import *
from musicautobot.music_transformer import *
from musicautobot.utils.midifile import *
from musicautobot.utils.file_processing import process_all


# num_tracks = [1, 2] # number of tracks to support
# cutoff = 5 # max instruments
# min_variation = 3 # minimum number of different midi notes played
# max_dur = 128
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./data/midi', help='Path to midi files')
    parser.add_argument('--data_savename', default='data_npy.pkl', help='name to save pkl file')
    parser.add_argument('--output_dir', default='../data/npy', help='path to save pkl file')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    