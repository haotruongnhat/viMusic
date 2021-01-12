from .markovify.chain import Chain, BEGIN, END
import pandas as pd
import music21,random,copy,math
from scipy.special import softmax
from collections import OrderedDict
import numpy as np
import copy
import re
import roman

DEFAULT_STATE_SIZE = 5
DEFAULT_NUM_CHORD_NEEDED = 4
DEFAULT_ATTENTION_TO_NUM_CHORDS = 2
DEFAULT_NUM_TARGET_CHORD_NEEDED = 4
DEFAULT_OFFSET_MEASURE = 3

class CPChain(object):
    """
    CPChain (Chord Progression Chain) is based on Markov Chain.
	"""
    def __init__(self, state_size=None):
        self._sheet = pd.read_csv('./archs/chord_analysis/chord_suggestion.csv')
        corpus = []
        for cp in self._sheet['roman_cp'].to_list():
        	corpus.append(cp.split(','))
        self.chain = Chain(corpus=corpus, state_size=5)
        # These chords can start the markov chains
        self.start_chord = list(list(self.chain.model.values())[0].keys())

        #If song is Major key `which is Cmaj`: (BEGIN, BEGIN, BEGIN, BEGIN, 'I') 
        #else minor key `which is Amin`: (BEGIN, BEGIN, BEGIN, BEGIN, 'vi')
        self.state_size = state_size or DEFAULT_STATE_SIZE

        self.refer_chord = []
        self.cp_length = [2]

        #TODO: need to modify this to learn dominant, ...
        self._c_major_chord_progression = ['I','ii','iii','IV','V','vi','vii°']
        self._a_minor_chord_progression = ['i','ii°','III','iv','v','VI','VII']

    def generate(self,extract_num_of_chord,key='C'): #or Am
        """
        Generate chord based on refer_chord
        - Input: Number of chord and key
        - Output: list of RomanNumeral
        """
        if extract_num_of_chord > len(self._refer_chord):
            raise ValueError("Unable to generate chord progression being longer than the designated refer chord")
        generated_chord_progression = [self._refer_chord[-1]] #last one must be the end of chord, heuristic
        current_idx = len(self._refer_chord) - 1
        for i in range(extract_num_of_chord - 1,0,-1):
            chord_idx = random.choice(list(range(i - 1,current_idx)))
            generated_chord_progression = [self._refer_chord[chord_idx]] + generated_chord_progression
            current_idx = chord_idx
        #convert all of them to music21.chord.Chord
        chords = []

        CM_to_Am = \
        {
            'I': 'III',
            'ii': 'iv',
            'iii': 'v',
            'IV': 'VI',
            'V': 'VII',
            'vi': 'i',
            'viii°': 'ii°'
        }

        keys = sorted(CM_to_Am.items(),key=lambda x: len(x[0]),reverse=True)
        CM_to_Am = OrderedDict(keys)

        for chord in copy.deepcopy(generated_chord_progression):
            #14/3/2020:  the sup, sup, &#9837, &deg and /
            is_flat = False
            numbers = re.findall(r'[0-9]+',chord)

            containSlash = 0
            for i in range(len(chord)):
                if chord[i] == '/' and i >= 1 and chord[i-1] != '<': 
                    containSlash=1
                    break

            if containSlash:
                chord = re.sub(r'/.*','',chord)
            if 'sup' in chord:
                chord = re.sub(r'<sup>([0-9]+)</sup>',r'/\1',chord)
            if 'sub' in chord:
                chord = re.sub(r'<sub>([0-9]+)</sub>',r'/\1',chord)
            if '&deg' in chord:
                chord = re.sub(r'&deg','//0',chord)
            if '&#9837' in chord: 
                chord = 'VII'
                is_flat = True
            if key == 'C':
                # try:
                #     index = self._c_major_chord_progression.index(chord)
                # except:
                #     #reverse key, then try again
                #     chord = chord.swapcase()
                #     index = self._c_major_chord_progression.index(chord)
                # result = self._c_major_chord_progression[index] + ('-' if is_flat else '')
                # chords.append(music21.roman.RomanNumeral(result,key))


                result = chord + ('-' if is_flat else '')
                chords.append(music21.roman.RomanNumeral(result,key))
            else:
                for i in CM_to_Am: 
                    if i in chord:
                        chord = re.sub(i,CM_to_Am[i],chord)
                        break

                # try:
                #     index = self._a_minor_chord_progression.index(chord)
                # except:
                #     #reverse key, then try again
                #     chord = chord.swapcase()
                #     index = self._a_minor_chord_progression.index(chord)
                # result = self._a_minor_chord_progression[index] + ('-' if is_flat else '')
                # chords.append(music21.roman.RomanNumeral(result,key))

                result = chord + ('-' if is_flat else '')
                chords.append(music21.roman.RomanNumeral(result,key))
        
            
        return chords
        # return generated_chord_progresion




if __name__ == '__main__':
    dump_suggestion = CPChain()
    print(dump_suggestion.generate(5,'Am'))
    print(dump_suggestion.generate(5,'C'))
