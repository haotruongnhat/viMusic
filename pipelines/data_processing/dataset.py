from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import os
import copy
import random
from sklearn.model_selection import train_test_split

class MusicDataset(Dataset):
    """
    This class using to load the preprocess pickle
    """    
    def __init__(self, pkl_file):
        self.samples = None
        self.load_preprocess_dataset(pkl_file)

    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def shape(self):
        return self.samples.shape
    
    def load_preprocess_dataset(self, file_path):
        """
        Load pre-process dataset
        
        Args:
            
            file_path (str): File path to pickle file
        """
        with open(str(Path(file_path)), 'rb') as handle:
            self.samples = pickle.load(handle)
            
class LyricsDataset(Dataset):
    """
    This class using to load the preprocess pickle
    """    
    def __init__(self,config,pkl_path=None,samples=None):
        self.config = config
        self.genre = config.genre
        self.ratio = config.ratio
        if pkl_path is not None and samples is not None:
            raise ValueError("Either only one of the attribute is not None")
        if pkl_path is None and samples is None:
            raise ValueError("There's nothingggg")
        if pkl_path is not None:
            self.samples = None
            self.load_preprocess_dataset(os.path.join(pkl_path,"wasabi_" + self.genre + ".pkl"))
        else:
            self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return
        {
            "input" : (2*w_size,max_ssm_size).
            "label" : [value] * 2 (value == 0 or value == 1)
        }
        """
        sample = self.samples[idx]
        label = np.zeros((sample['label'].shape[0],2))
        for i,idx in enumerate(sample['label']):
            label[i][idx] = 1
        return {
            "input" : sample['input'],
            "label" : label
        }

    @property
    def shape(self):
        return [self.samples[0]['input'].shape,self.samples[0]['label'].shape]
    
    def load_preprocess_dataset(self, file_path):
        """
        Load pre-process dataset
        
        Args:
            
            file_path (str): File path to pickle file
        """
        with open(file_path, 'rb') as handle:
            self.samples = pickle.load(handle)

    def split_dataset(self):
        """
        return a train set and a test set
        """
        if self.config.train_shuffle:
            random.shuffle(self.samples)
        return (
            LyricsDataset(self.config,samples=self.samples[:int(len(self.samples) * self.ratio)]),
            LyricsDataset(self.config,samples=self.samples[int(len(self.samples) * self.ratio):])) 