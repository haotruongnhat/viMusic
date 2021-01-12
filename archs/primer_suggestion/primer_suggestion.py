import pandas as pd
import os, sys
from scipy.spatial import distance
import numpy as np
import math
import ast
import random
import miditoolkit
import pretty_midi
import pymongo
import pickle
from sshtunnel import SSHTunnelForwarder
import copyreg
import ast
import music21

from common.music_item import ItemProcessor
from common.object import Note, Chord
from common.constants import CONVERSION, CHORD

from miditoolkit.midi.parser import MidiFile 

from common.vocab import RemiVocabItem
from common.vocab_definition import REMI

class PrimerSuggestion(object):
    """
    Suggest primer melody for the model
    - Input : Mood and metadata
    - Output: a list of token
    """
    #CONSTANT FOR MONGO ACCESSIBILITY
    SERVER_URL = "viws.ddns.net"
    REMOTE_ADDRESS = '127.0.0.1'
    REMOTE_PORT = 27017
    USER_NAME = "vimusic" #tam
    PASSWORD = "vimusic1" #tam
    DATABASE_NAME = "vimusic"
    PRIMER_MELODY_COLLECTION = "vimusic_primer_melody"

    def __init__(self):
        self._vocab = RemiVocabItem(REMI.INDEX_TOKENS)
        #begin server
        #1. Connect to the mongo database
        self._server = SSHTunnelForwarder(
        PrimerSuggestion.SERVER_URL,
        ssh_username=PrimerSuggestion.USER_NAME,
        ssh_password=PrimerSuggestion.PASSWORD,
        remote_bind_address=(PrimerSuggestion.REMOTE_ADDRESS,PrimerSuggestion.REMOTE_PORT)
        )
        
        self._server.start()
        #Get database
        self._db = pymongo.MongoClient(self._server.local_bind_host,self._server.local_bind_port)
        collection = self._db[PrimerSuggestion.DATABASE_NAME][PrimerSuggestion.PRIMER_MELODY_COLLECTION]

        #primer collection
        self._primer = self._db[PrimerSuggestion.DATABASE_NAME][PrimerSuggestion.PRIMER_MELODY_COLLECTION]

    def _closeness(self,x,
    energy=0.0,
    speechiness=0.0,
    acousticness=0.0,
    instrumentalness=0.0,
    liveness=0.0,
    valence=0.0):
        #energy
        en = math.pow(x['energy'] - energy,2)
        spe = math.pow(x['speechiness'] - speechiness,2)
        aco = math.pow(x['acousticness'] - acousticness,2)
        ins = math.pow(x['instrumentalness'] - instrumentalness,2)
        li = math.pow(x['liveness'] - liveness,2)
        va = math.pow(x['valence'] - valence,2)
        return math.sqrt(en + spe + aco + ins + li + va)

    def suggest(self,danceability=0.0,
    energy=0.0,
    speechiness=0.0,
    acousticness=0.0,
    instrumentalness=0.0,
    liveness=0.0,
    valence=0.0,
    tonic='C', #or A
    mode='Major', #or minor (it should be CMajor, or Aminor)
    numerator=4,
    denominator=4,file_path=None):
        #filter by key and time signature
        objects = list(self._primer.find({
            'tonic' : tonic,
            'mode' : mode,
            'numerator' : numerator,
            'denominator' : denominator,
        }))

        if len(objects) == 0:
            print("No primer melody found")
            return None

        #sort object based on closeness the moods
        objects = sorted(objects, key=lambda x : self._closeness(
            x,
            energy=0.0,
            speechiness=0.0,
            acousticness=0.0,
            instrumentalness=0.0,
            liveness=0.0,
            valence=0.0,
        ),reverse = True)

        choice = random.choice(objects[:int(len(objects) / 3)])
        melody = pickle.loads(choice['primer_melody_data'])

        #get list of notes
        notes = []
        for instrument in melody.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append(note)

        #Convert to system's note
        system_notes = []

        for note in notes:
            start_duration = CONVERSION.ticks_to_duration(note.start,
            melody.ticks_per_beat)
            duration = CONVERSION.ticks_to_duration(note.end,
            melody.ticks_per_beat) - start_duration
            pitch = note.pitch
            velocity = note.velocity
            system_notes.append(Note(start=start_duration,duration=duration,pitch=pitch,velocity=velocity))
    
        #getting things ready
        system_notes = sorted(system_notes,key=lambda x : x.start)
        chord_start = min(system_notes,key=lambda x: x.start)
        chord_start = chord_start.start
        chord_duration = max(system_notes,key=lambda x : x.start + x.duration)
        chord_duration = chord_duration.start + chord_duration.duration
        pitch = CHORD.PITCH_NAME.index(choice['chord'])
        quality_index = CHORD.FULL_QUALITY_ABBR.index(choice['chord_mode'])
        quality = CHORD.FULL_QUALITY[quality_index]
        system_chord = [Chord(start=chord_start,
            pitch=pitch,
            quality=quality,
            duration=chord_duration)]
        time_signature = music21.meter.TimeSignature("{}/{}".format(
            choice['numerator'],
            choice['denominator']
        ))

        result = ItemProcessor.events_to_tokens(system_notes,system_chord,
        time_signature,
        self._vocab)
        result = result[:np.where(result == 1)[0][0]]

        #pickle the result
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

        return result
        
    def stop(self):
        self._db.close()
        self._server.stop()

if __name__ == '__main__':
    test  = PrimerSuggestion()
    tokens = test.suggest(tonic='A',mode='Minor')
    test.stop()
    
