import  sys, os 
import argparse
import numpy as np
from MTransformer.numpy_encode import *
from MTransformer.utils.file_processing import process_all, process_file
from MTransformer.config import *
from MTransformer.music_transformer import *
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Generate midi file')
    parser.add_argument('--output_dir', default='../data/npy', help='path to store generate midi files')
    parser.add_argument('--model_path', default='data/cached/models/MusicTransformer.pth', help='path to .pth model file')
    parser.add_argument('--mode', default='gfm', help='choose generate  mode: ["random", "gfm"], default: gfm')
    parser.add_argument('--n_words', default=600, help='length of generate music')
    parser.add_argument('--encode_position', default=False, type=bool)
    parser.add_argument('--num_outputs', default=3, type=int, help='Number of output midi')
    args = parser.parse_args()

    midi_path =  str(Path('./data/midi/examples'))
    data_path = str(Path('./data/cached'))

    data = MusicDataBunch.empty(data_path)
    vocab = data.vocab

    ### params
    n_words = int(args.n_words)
    num_outputs = args.num_outputs
    mode = args.mode 
    pretrained_path = args.model_path
    encode_position = args.encode_position
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    config = default_config()
    config['encode_position'] = encode_position
    learn = music_model_learner(data, pretrained_path=pretrained_path, config = config.copy())

    midi_files = get_files(midi_path, recurse=True, extensions='.mid')
    # idx = np.random.randint(0,len(midi_files))
    
    cutoff_beat = 10
    

    
    arr_note_temp = np.arange(0.1,1.9,step=0.1)
    arr_dur_temp = np.arange(0.5,1.5,step=0.1)
    note_temp = 1.8 # Determines amount of variation in note pitches
    dur_temp = 1.2 # Amount of randomness in rhythm
    top_k = 25
    np.random.seed(1)

    if mode == 'random':
        empty_item = MusicItem.empty(vocab)
        for i in num_outputs:
            note_temp = np.random.choice(arr_note_temp)
            dur_temp = np.random.choice(arr_dur_temp)
            pred, full = learn.predict(empty_item, n_words=n_words, temperatures=(note_temp, dur_temp), min_bars=12, top_k=top_k, top_p=0.7)
            name_to_save = 'MT_random_generate_midi_{}_{}'.format(str(datetime.now().time()), i+1)
            path_to_save = os.path.join(args.output_dir, name_to_save)
            pred.save_to_mid(path_to_save)
    # Generate midi file from midi 
    elif mode == 'gfm':
        for i in num_outputs:
            note_temp = np.random.choice(arr_note_temp)
            dur_temp = np.random.choice(arr_dur_temp)
            f = np.random.choice(midi_files)
            item = MusicItem.from_file(f, data.vocab)
            seed_item = item.trim_to_beat(cutoff_beat)
            pred, full = learn.predict(seed_item, n_words=n_words, temperatures=(note_temp, dur_temp), min_bars=12, top_k=top_k, top_p=0.7)
            name_to_save = 'MT_gfm_generate_midi_{}_{}'.format(str(datetime.now().time()), i+1)
            path_to_save = os.path.join(args.output_dir, name_to_save)
            full_song = seed_item.append(pred)
            full_song.save_to_mid(path_to_save)
       