import prosodic #li to identify stress
from finnsyll import FinnSyll
import multiprocessing
import collections
import re
import torch
import random
import copy
import json
from py3nvml.py3nvml import *

from archs.lyrics_analysis.lyrics_segmentation.generator import Segmentator
from archs.lyrics_analysis.semantic_analysis.cluster import lyrics_cluster

from pipelines.data_processing.configs import pop_lyrics_analysis_config, rock_lyrics_analysis_config, electro_lyrics_analysis_config

from server.workers.cuda_worker import CudaWorker
from archs.voice_synth.voice_interface import word_to_syllable_phonemes

class LyricsExtractionWorker(CudaWorker):
    """
    An Unorderedworker working with lyrics analysis
    """
    def __init__(self,
        model,
        allow_memory,
        global_lock=None,
        pool=None
        ):

        super(LyricsExtractionWorker,self).__init__(global_lock,pool,allow_memory)
        
        self._genre = model[0]
        self._model_path = model[1]
        if self._genre == "pop":
            self._seg_config = pop_lyrics_analysis_config
        elif self._genre == 'rock':
            self._seg_config = rock_lyrics_analysis_config
        elif self._genre == 'electro':
            self._seg_config = electro_lyrics_analysis_config
        

    def doInit(self):
        #Alocate GPU memory
        self.acquire()
        #Backup syllable splitter
        self._backup_syllables_splitter = FinnSyll()
        self.init_cuda_memory()
        print("A {} LyricsExtractor Model is using GPU {}".format(self._genre,self._cuda_device_index))
        #Randomize output of different songs genre
        torch.manual_seed(100000)
        self._segmentator = Segmentator(
            config=self._seg_config,
            checkpoint_path = self._model_path,
            cuda_index=self._cuda_device_index)
        self.release()

    def doTask(self,query):
        """
        query = {
            "time_signature_numerator" : int,
            "time_signature_denominator" : int,
            "key_signature" : string,
            "bpm" : string,   
            "danceability" : float,
            "energy" : float,
            "speechiness" : float,
            "valence" : float,
            "acousticness" : float,
            "instrumentalness" : float,
            "liveness" : float,
            "status" : string,
            "idProcess" : string,
            "percentage" : int,
            "option" : int,
            "lyric" : (input, string),
            "path" : string,
            "result" : {},
            "mood_analysis" : [],
            "option" : int
            "{}_stresses" : (output,matrix of int), with {} = self._genre,
            "{}_syllables" : (output, matrix of string), with {} = self._genre,
            "{}_section_groups": (output,dictionary of segments), with {} = self._genre,
            "{}_boundaries" : (output, list of int denoting end of segments), with {} = self._genre,
        })
        """
        if query is None:
            return None
        lyrics = collections.defaultdict(list)
        syllables = collections.defaultdict(list)

        #Get boundaries
        boundaries = self._segmentator.predict(query["lyric"])
        if boundaries is None:
            boundaries = []

        #Get segments and group sections
        unprocessed_segments,section_groups = lyrics_cluster(query["lyric"],boundaries)
        
        for e,segment in enumerate(unprocessed_segments):
            for sentence in segment:
                #normalizing whitespaces into single space
                l_norm = re.sub(r'[\t\ ]+', ' ', sentence)

                #remove special characters
                l_spec = re.sub(r"[^A-Za-z.!?\'\ ]*",'',l_norm)

                #trim sentences
                l_trim = l_spec.strip()

                #Split to words
                words = [x for x in re.split(' |- ',l_trim) if x != '']

                original_syllables = []
                stress_list = []

                for word in words:
                    try:
                        ss, stresses = word_to_syllable_phonemes(word)
                    except:
                        print("Error with lyrics")
                        query['status'] = 'fail'
                        return query
                    stress_list = stress_list + stresses
                    original_syllables = original_syllables + ss

                #append to the section
                lyrics[e].append(stress_list)
                syllables[e].append(original_syllables)

        #Merge back to lyrics
        query["{}_stresses".format(self._genre)] = lyrics
        query["{}_syllables".format(self._genre)] = syllables
        query["{}_section_groups".format(self._genre)] = section_groups
        query["{}_boundaries".format(self._genre)] = boundaries

        #Send to redis to update progress
        query["percentage"] += 10
        self.update_progress(query)

        print("{}: LyricsExtraction done".format(query["idProcess"]))

        return query

        
