from abc import ABCMeta, abstractmethod, ABC
import abc
import random
import os
import pandas as pd
from pathlib import Path
import pickle
import math
import numpy as np
import lyricwikia
import ast
from tqdm import tqdm
from glob import glob

class DatasetTransform(ABC):

    def __init__(self, config):
        #Intialize basic configuration
        self._dataset_dir = config.dataset_dir
        self._file_handler= config.handler
        self._data_path = config.data_path
        self._num_workers = config.num_workers
        self._samples = None

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self):
        pass
    
    @abc.abstractmethod
    def preprocess_dataset(self,num_workers=1):
        pass

    def save_preprocess_dataset(self):
        """
        Pre-process the whole dataset and save the samples
    
        Args:

            output_file (str): File path to pickle file
            save_existing_samples (bool): If True, this will save the existing processed samples.
                                        If samples is empty, then pre-process the whole dataset
                                        If False, pre-process the dataset and save that samples
        """
        if self._samples is None:
            self.preprocess_dataset(self._num_workers)
            
        with open(str(self._data_path), 'wb') as handle:
            pickle.dump(self._samples, handle, protocol=pickle.HIGHEST_PROTOCOL)


class MusicDatasetTransform(DatasetTransform):
    """
    Class to  manage and pre-process the music data
    
    Args:
    
        dataset_dir (str): Folder path to music score (xml) file
        transform (torchvision.transforms.Compose): Contains several transform pipeline stored into composed pipeline
        
    Return:
    
        pre-processed samples
    """
    def __init__(self, config):
        super(MusicDatasetTransform,self).__init__(config)

        self._vocab = config.vocab
        self._transform = config.transform_pipeline

        self._file_extensions = config.file_extensions

        self._file_list = [file_obj for ext in self._file_extensions for file_obj in list(glob('{}/*.{}'.format(self._dataset_dir,ext)))]

        self._seq_len = config.seq_len
        
        
    def __len__(self):
        return len(self._file_list)
        
    def __getitem__(self, idx):
        # Read the score file
        # import pdb; pdb.set_trace()
        sample = None
        # Data pre-processing
        item = self._file_handler(path=str(self._file_list[idx]),vocab = self._vocab)
        if self._transform:
            sample = self._transform(item)

        return sample
    
    def preprocess_dataset(self, num_workers=1):
        """
        Iterate through all the files and doing the transform
        """
        # Get the sample shape
        single_sample_shape = None
        for idx in range(len(self)):
            try:
                sample = self[idx]
                single_sample_shape = sample[0].shape
                break
            except:
                pass
        samples = np.zeros((len(self),), dtype=object)
        valid_sample_indices = np.zeros(len(self), dtype=bool)
        
        inputs = tqdm(range(len(self)), desc="|-Pre-process dataset-|")
    
        def get_sample(idx):
            try:
                samples[idx] = self[idx]
                valid_sample_indices[idx] = True
            except Exception as e:
                pass
                # print('\n Error: {}. \n File name: {}'.format(str(e), self.file_list[idx]))
        for i in inputs:
            get_sample(i)
        samples = samples[valid_sample_indices]
        new_shape = (samples.shape[0] * samples[0].shape[0],) # Number of files * number of transposed
        samples = samples.reshape(new_shape)

        samples_split = []
        #Split
        for sample in samples:
            sample = np.squeeze(sample,0)
            for i in range(0,sample.shape[0],self._seq_len):
                if i + self._seq_len > sample.shape[0]:
                    short_sample = sample[i : sample.shape[0]]
                    pad_value = self._seq_len - short_sample.shape[0]
                    if pad_value > short_sample.shape[0] / 2:
                        print("Skip short sequences")
                        continue
                    pad_sample = np.pad(short_sample,(0,pad_value),constant_values=self._vocab.pad_idx)
                    samples_split.append(pad_sample)
                else:
                    samples_split.append(sample[i : i + self._seq_len])
        samples_split = np.asarray(samples_split)
        self._samples = samples_split
        print('Samples generated. Samples size: {}, length: {}'.format(self._samples.shape[0], self._samples.shape[-1]))

class LyricsDatasetTransform(DatasetTransform):
    """
    Class to preprocess lyrics data into w_size segments

    Args:
        dataset_dir (str): folder to dataset contianing lyrics
        transform (torchvision.transforms.Compose)

    Return:
        Pre-process dataset
    """
    def __init__(self,config):
        super(LyricsDatasetTransform,self).__init__(config)

        self._max_parts = config.max_parts
        self._max_songs = config.max_songs
        self._shuffle = config.shuffle
        self._transform = config.transform_pipeline
        self._genre = config.genre

        self._obj_list = dict()
        #read ssm data
        for i in range(self._max_parts):
            try:
                ssm = pd.read_pickle(os.path.join(
                    self._dataset_dir,
                    "ssm_wasabi",
                    "ssm_wasabi_{}.pickle".format(i)))
                self._obj_list.update(ssm)
            except:
                print("Corrupted ssm_wasabi_{}.pickle. Skip.".format(i))

        if len(self._obj_list) == 0:
            raise RuntimeError("All ssm file corrupted.")

        #add keys to dict and convert to list
        for key in self._obj_list:
            self._obj_list[key]["id"] = key
        self._obj_list = list(self._obj_list.values())

        #read wasabi songs
        self._wasabi_info = pd.read_csv(os.path.join(
            self._dataset_dir,
            "wasabi_songs.csv"
        ),sep="\t")

        if self._max_songs > len(self._obj_list):
            print("Max song exceeds number of obj. resetting max songs")
            self._max_songs = len(self._obj_list)

    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self,idx):
        """
        Output
        {
            "input" : (2*,max_ssm_size)
            "label" : int
        }
        """
        if self._shuffle:
            self._obj_list = random.shuffle(self._obj_list)

        #Check if id of idx exist
        obj = self._obj_list[idx]
        id = obj['id'] #id
        ssm = obj['line'] #numpy matrix

        segment_ssm = obj['segment'] #numpy segment matrix
        song_row = self._wasabi_info[self._wasabi_info['_id'] == id]

        if len(song_row) > 1:
            print("Warning: More than one row information. Choose randomly")
            info_idx = random.choice(range(len(song_row)))
        elif len(song_row) == 1:
            info_idx = 0
        else:
            print("No information found. Return None")
            return None

        artist = song_row['artist'].values[info_idx]
        song = song_row['title'].values[info_idx]
        genre = song_row['genre'].values[info_idx]
        
        if isinstance(genre,str):
            genre = ast.literal_eval(genre)
        else:
            return None
            

        if not any([self._genre in x for x in genre]):
            return None

        try:
            item = self._file_handler(
                id,
                artist,
                song,
                genre,
                ssm,
                segment_ssm)
        except:
            print("Something wrong with this id. Returning None.")
            return None

        if self._transform:
            samples = self._transform(item)
            if samples is None:
                return None

            input = np.asarray([x['input'] for x in samples])

            label = np.asarray([x['label'] for x in samples])

            return {
                "input" : input,
                "label" : label
            }
    
        return None

    def preprocess_dataset(self,num_workers=1):
        """
        Iterating through all the files and do the transformation
        Output:
        A list of dictionary
        """
        # Get the sample shape
        self._samples = []
        i = 0
        iterator = 0
        while iterator < len(self._obj_list):
            samples = self[iterator]
            if samples is not None: 
                i = i + 1
                self._samples.append(samples)
                print(i)
            if i == self._max_songs:
                break
            iterator = iterator + 1
               

               
            
