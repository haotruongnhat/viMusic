import signal
import mpipe
import torch
import redis
import collections
import numpy as np
import copy
import uuid
import json
import os
import pickle
import random
import math
import copy
import torch

from common.music_item import ItemProcessor
from common.vocab import RemiVocabItem
from common.music_item import RemiItem
from common.vocab_definition import REMI
from common.func import otsu_melody

from common.music_item import ItemProcessor
from common.object import Note, Chord
from common.constants import CONVERSION, CHORD

from archs.instruments.generate import mastering, OUTPUT_TYPE
from archs.voice_synth.voice_interface import VoiceDNN, word_to_syllable_phonemes

from server.workers.cuda_worker import CudaWorker

class RemiAndMasteringWorker(CudaWorker):
    """
    A Worker for the generation
    """

    def __init__(self,
        model,
        allow_memory,
        global_lock=None,
        pool=None,
        ):
        super(RemiAndMasteringWorker,self).__init__(global_lock,pool,allow_memory)

        self._genre = model[0]
        self._model_path = model[1]
        self._vocab = RemiVocabItem(REMI.INDEX_TOKENS)

    def doInit(self):
        """
        Initialize cuda
        """
        self.acquire()
        self.init_cuda_memory()

        print("A {} REMI worker is using GPU {}".format(self._genre,self._cuda_device_index))
        self._melody_generator = torch.load(
            self._model_path, map_location='cuda:{}'.format(self._cuda_device_index)) \
            .to('cuda:{}'.format(self._cuda_device_index))
        
        self._voice_generator = VoiceDNN(self._cuda_device_index)
        self.mastering = mastering

        self.release()

    def doTask(self,query):
        """
        query = {
            "time_signature_numerator" : (input,int),
            "time_signature_denominator" : (intput,int),
            "key_signature" :(input,string),
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
            "option" : (input,int),
            "lyric" : string,
            "path" : string,
            "result" : (output,{}),
            "{}_stresses" : matrix of string, with {} = self._genre,
            "{}_syllables" : matrix of string, with {} = self._genre,
            "{}_section_groups": dictionary of segments, with {} = self._genre,
            "{}_boundaries" : list of int denoting end of segments, with {} = self._genre,
            "primer_path" : (input,string)
        })
        """
        if query['status'] == 'fail':
            return query
        if query is None:
            return None

        #Pickle the primer tokens
        with open(query["primer_path"], 'rb') as f:
            primer_tokens = pickle.load(f)

        use_token = None
        full_song = []

        #Structure the song 
        melody_dict = collections.defaultdict(dict)
        #Order melody according to smallest section
        reverse_structure_smallest_label = {min(v) : k for k,v in query['{}_section_groups'.format(self._genre)].items()}
        order_structure_id = [reverse_structure_smallest_label[k] for k in sorted(list(reverse_structure_smallest_label.keys()))]
        #Generate melody for each sections
        use_token = primer_tokens.tolist()
        last_length = 1
        for turn,structure_id in enumerate(order_structure_id):
            #Finding in maximum sentence, and the maximum syllable of each structure
            max_num_of_sentence = max([len(query['{}_stresses'.format(self._genre)][i]) for i in query["{}_section_groups".format(self._genre)][structure_id]])
            max_num_sylls = [0] * max_num_of_sentence
            for section_id in query["{}_section_groups".format(self._genre)][structure_id]:
                for sen_id,sentence in enumerate(query['{}_stresses'.format(self._genre)][section_id]):
                    if len(sentence) > max_num_sylls[sen_id]:
                        max_num_sylls[sen_id] = len(sentence)
            #Begin generate
            for sen_id,num_sylls in enumerate(max_num_sylls):
                use_token = self._melody_generator.generate(
                    self._vocab,
                    num_sylls,
                    primer_tokens = use_token,
                    temperature_range=(0.5,10.5) if turn % 2 == 0 else (0.8,2.0),
                    device = 'cuda:{}'.format(self._cuda_device_index),
                    beam_width=3, #best
                    look_back=15)
                cut_token = use_token[last_length - 1:] #Keeping 0
                last_length = len(use_token)
                bar_seq = ItemProcessor.tokens_to_barsequence(np.asarray(cut_token),self._vocab)
                #store to melody dict
                melody_dict[structure_id][sen_id] = bar_seq

        #Lyrics alginment for each sentence
        reverse_section_groups = dict()
        del use_token
        for key in query["{}_section_groups".format(self._genre)]:
            for section_id in query["{}_section_groups".format(self._genre)][key]:
                reverse_section_groups[section_id] = key

        for section_id in query['{}_stresses'.format(self._genre)]:
            for sen_id,sentence_stresses in enumerate(query['{}_stresses'.format(self._genre)][section_id]):
                structure_id = reverse_section_groups[section_id]
                bar_seq = copy.deepcopy(melody_dict[structure_id][sen_id])
                for i,_ in enumerate(bar_seq):
                    bar_seq[i] = sorted(bar_seq[i],key= lambda x : x.start)

                _,strong_beat_index,weak_beat_index = otsu_melody(bar_seq)
                highest_tuple_index = set()
                highest_tuple_index.update(strong_beat_index)
                highest_tuple_index.update(weak_beat_index)
                highest_tuple_index = sorted(highest_tuple_index)

                # min_cost = float('inf')
                # random.seed(1000)
                best_merger = highest_tuple_index[:len(sentence_stresses)]
                # superset = sorted(random.sample(list(highest_tuple_index),k=))
                # #calculate distance between 2 consecutive note: must be normalized
                # dist_costs = []
                # dur_per_bar = CONVERSION.bar_to_duration_ratio()
                # for i in range(1,len(superset)):
                #     bar_cost = dur_per_bar * (superset[i][0] - superset[i - 1][0])
                #     note_cost = bar_seq[superset[i][0]][superset[i][1]].duration - bar_seq[superset[i - 1][0]][superset[i - 1][1]].duration
                #     dist_costs.append(bar_cost + note_cost)
                # if len(dist_costs) == 0:
                #     mean_dist_cost = 0.0
                #     dist_cost = 0.0
                # else:
                #     mean_dist_cost = float(sum(dist_costs)) / len(dist_costs)
                #     dist_cost = sum([math.pow(x - mean_dist_cost,2) for x in dist_costs]) / len(dist_costs)
                # dist_cost = dist_cost * 4

                # beat_cost = 0.0
                # for i,syllable_value in enumerate(sentence_stresses):
                #     stress_value = 0
                #     note_value = 1 if superset[i] in strong_beat_index else 0
                #     beat_cost += abs(stress_value - note_value)
                #     beat_cost = float(beat_cost) / len(sentence_stresses) #down sampling to 0 to 1
                #     if min_cost > beat_cost * dist_cost:
                #         min_cost = beat_cost * dist_cost
                #         best_merger = superset

                flatten_syllables = query['{}_syllables'.format(self._genre)][section_id][sen_id]

                ###Merge lyrics with best
                for i,ti in enumerate(best_merger):
                    bar_seq[ti[0]][ti[1]].syllable = flatten_syllables[i] if flatten_syllables[i] is not None else ""

                for bar in bar_seq:
                    full_song.append(bar)

        raw_name = str(uuid.uuid4())
        wav_name = raw_name + ".wav"
        wav_path = os.path.join(query['path'],wav_name)

        score = ItemProcessor.barseq_to_score(full_song, False)
        #Debugging purposes
        score.write("xml",fp="./.cache/.result/{}.xml".format(raw_name))

        output_type = OUTPUT_TYPE.WAV_FILE
        print(raw_name, self._genre, query['option'], output_type, wav_path)
        try:
            duration = mastering(score,self._genre,query['option'], output_type,wav_path, self._voice_generator, 90)
        except:
            if query['option'] != 0:
                print("Error with option 3, running option 0")
                duration = mastering(score,self._genre,0, output_type,wav_path, self._voice_generator, 90)
        ####
        if duration is None:
            duration = 0
        query['result'][self._genre] = {
            "idMusic" : raw_name,
            "nameMusic" : wav_name,
            "duration" : int(duration)
        }

        #Send to redis to update progress
        query["percentage"] += 20
        self.update_progress(query)

        print("{}: RemiAndMastering done".format(query["idProcess"]))

        return query
