import os 

from MTransformer.numpy_encode import *
from MTransformer.config import *
from MTransformer.music_transformer import *
from MTransformer.utils.midifile import *
from MTransformer.utils.file_processing import process_all


# num_tracks = [1, 2] # number of tracks to support
# cutoff = 5 # max instruments
# min_variation = 3 # minimum number of different midi notes played
# max_dur = 128
def timeout_func(data, seconds):
    print("Timeout:", seconds, data.get('midi'))

def process_metadata(midi_file):
    # Get outfile and check if it exists
    out_file = numpy_path/midi_file.relative_to(midi_path).with_suffix('.npy')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists(): return
    
    npenc = transform_midi(midi_file)
    if npenc is not None: np.save(out_file, npenc)

def transform_midi(midi_file):
    input_path = midi_file
    
    # Part 1: Filter out midi tracks (drums, repetitive instruments, etc.)
    try: 
#         if duet_only and num_piano_tracks(input_path) not in [1, 2]: return None
        input_file = compress_midi_file(input_path, min_variation=3, cutoff=5) # remove non note tracks and standardize instruments
        
        if input_file is None: return None
    except Exception as e:
        if 'badly form' in str(e): return None # ignore badly formatted midi errors
        if 'out of range' in str(e): return None # ignore badly formatted midi errors
        print('Error parsing midi', input_path, e)
        return None
        
    # Part 2. Compress rests and long notes
    stream = file2stream(input_file) # 1.
    try:
        chordarr = stream2chordarr(stream) # 2. max_dur = quarter_len * sample_freq (4). 128 = 8 bars
    except Exception as e:
        print('Could not encode to chordarr:', input_path, e)
        print(traceback.format_exc())
        return None
    
    # Part 3. Compress song rests - Don't want songs with really long pauses 
    # (this happens because we filter out midi tracks).
    chord_trim = trim_chordarr_rests(chordarr)
    chord_short = shorten_chordarr_rests(chord_trim)
    delta_trim = chord_trim.shape[0] - chord_short.shape[0]
#     if delta_trim > 500: 
#         print(f'Removed {delta_trim} rests from {input_path}. Skipping song')
#         return None
    chordarr = chord_short
    
    # Part 3. Chord array to numpy
    npenc = chordarr2npenc(chordarr)
    if not is_valid_npenc(npenc, input_path=input_path):
        return None
    
    return npenc
    

def create_databunch(files, data_save_name, path):
    save_file = os.path.join(path, data_save_name)
    if os.path.exists(save_file):
        data = load_data(path, data_save_name)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        vocab = MusicVocab.create()
        processors = [OpenNPFileProcessor(), MusicItemProcessor()]

        data = MusicDataBunch.from_files(files, path, processors=processors, encode_position=True)
        data.save(data_save_name)
    return data

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./data/midi', help='Path to midi files')
    parser.add_argument('--data_path', default='data_npy.pkl', help='name to save pkl file')
    parser.add_argument('--output_dir', default='../data/npy', help='path to save pkl file')

    args = parser.parse_args()

    # midi_path = '/home/tony/Vimusic/dataset/BigPackMIDI'
    midi_path = args.input_dir
    # Location of preprocessed numpy files
    # numpy_path = '/home/tony/Vimusic/dataset_tranformer/EDM/EDM_npy'
    numpy_path = args.output_dir 
    # Location of models and cached dataset
    data_path = '/home/tony/dataset_tranformer/EDM/cached'
    data_path = args.data_path
    data_save_name = 'classical_data_save.pkl'
    cutoff = 5 # max instruments
    min_variation = 3 
    print('---Test---')
    if not os.path.isdir(numpy_path):
        os.makedirs(numpy_path)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    midi_files = get_files(midi_path, '.midi', recurse=True)
    print(len(midi_files))
    processed = process_all(process_metadata, midi_files, timeout=120)
    numpy_files = get_files(numpy_path, extensions='.npy', recurse=True)
    create_databunch(numpy_files, data_save_name=data_save_name, path=data_path)

    # batch_size = 16
    # encode_position = True
    # dl_tfms = [batch_position_tfm] if encode_position else []
    # data = load_data(data_path, data_save_name, bs=batch_size, encode_position=encode_position, dl_tfms=dl_tfms)