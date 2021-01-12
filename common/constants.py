from enum import Enum
from music21 import *
from math import *

SEQType = Enum('SEQType', 'Mask, Sentence, Melody, Chords, Empty')


class CONVERSION:
    def quarterLength_to_duration(value):
        return int(value * 4)

    def duration_to_quarterLength(value):
        return value / 4.0

    def ticks_to_duration(value,ticks_per_beat=480):
        beats = float(value) / ticks_per_beat
        return CONVERSION.quarterLength_to_duration(beats)
    
    def bar_to_duration_ratio(time:meter.TimeSignature = meter.TimeSignature('4/4')):
        how_many_beat_per_bar = time.numerator
        
        # Calculation: Time Sig = 6/8
        #### Each beat in measure is eighth note = 0.5 quarterLength
        #### Duration = 4 * quarterLength = 2
        duration_per_beat = CONVERSION.quarterLength_to_duration( 4 * (1 / time.denominator) )
        
        return duration_per_beat * how_many_beat_per_bar
       
    def bar_to_quarterLength_ratio(time:meter.TimeSignature = meter.TimeSignature('4/4')):
        how_many_beat_per_bar = time.numerator
        
        # Calculation: Time Sig = 6/8
        #### Each beat in measure is eighth note = 0.5 quarterLength
        #### Duration = 4 * quarterLength = 2
        quarterLength_per_beat = 4 * (1 / time.denominator)
        
        return quarterLength_per_beat * how_many_beat_per_bar
       
    def duration_to_bar_ratio(time:meter.TimeSignature = meter.TimeSignature('4/4')):
        return 1 / CONVERSION.bar_to_duration_ratio(time)

    def quarterLength_to_bar_ratio(time:meter.TimeSignature = meter.TimeSignature('4/4')):
        return 1 / CONVERSION.bar_to_quarterLength_ratio(time)

class DEF:
    VALTSEP = -1 # separator value for numpy encoding
    VALTCONT = -2 # numpy value for TCONT - needed for compressing chord array

    BEAT_PER_BAR = 4
    
    NORMAL_RANGE = (0, 127)
    PIANO_RANGE = (21, 108)
    
    MELODY_PART_INDEX = 0
    CHORD_PART_INDEX = 1
    
    TOKENS_INDEX = 0
    POSITIONAL_ENC_INDEX = 1
    SHIFTED_TOKENS_INDEX = 2

class CHORD:
    PITCH_CLASSES = list(range(0,12))
    PITCH_NAME = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    SIMPLE_QUALITY = ['maj', 'min']
    FULL_QUALITY = ['major', 'minor', 'diminished', 'augmented', 'dominant']
    FULL_QUALITY_ABBR = ['maj','min','dim','aug','dom']

class SIZE:
    NOTE = 128 #128 midi pitches
    VELOCITY = 128

    DURATION = CONVERSION.bar_to_duration_ratio()* 2

    FULL_CHORD = len(CHORD.PITCH_NAME) * len(CHORD.FULL_QUALITY)
    SIMPLE_CHORD = len(CHORD.PITCH_NAME) * len(CHORD.SIMPLE_QUALITY)

    MAX_NOTE_DURATION = CONVERSION.quarterLength_to_duration(8*DEF.BEAT_PER_BAR)

MAX_MIDI_PITCH = 127
MIN_MIDI_PITCH = 0

class COMPOSER:

    POP = 'pop'

    CLASSICAL = 'classical'

    ROCK = 'rock'

    ELECTRO = 'electro'

    SONG_SECTIONS = {
        'intro' : 0,
        'verse' : 1,
        'pre_chorus' : 2,
        'chorus' : 3,
        'bridge' : 4,
        'outro' : 5
    }
    # 1 sentence = 4 measure
    # Number of sentence in a section
    DEFAULT_SONG_STRUCTURE = {
        "intro" : [1,2,4], #random
        "verse" : 4,
        "pre_chorus" : 1,
        "chorus" : 8, #the two half 4 repeat each other -> generate for 4 but duplicate for whole chorus
        "bridge" : 2 #not know what is it but... meh ...
    }
    GENRE = {
        'pop' : 6,
        'jazz': 7,
        'classical' : 8,
        'rock' : 9
    }

    MOOD = {
        'happy' : 10,
        'sad' : 11
    }

    ROMAN_CHORD_PROGRESSION = {
        'c_major' : ['I','ii','iii','IV','V','vi','vii°'],
        'a_minor' : ['i','ii°','III','iv','v','VI','VII']
    }

    CONTRACTION_MAP = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"}