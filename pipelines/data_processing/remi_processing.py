import copy
from pathlib import Path
import os, sys
sys.path[0] = os.getcwd()

from pipelines.data_processing.transform import MusicDatasetTransform
# from pipelines.data_processing.configs import remi_no_sharp_with_syllable_config
from pipelines.data_processing.configs import remi_no_sharp_config

dataset_transform = MusicDatasetTransform(remi_no_sharp_config)
dataset_transform.save_preprocess_dataset()
