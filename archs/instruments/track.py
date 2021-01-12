from common.object import Configs
import music21
import copy
import random
from common.constants import CONVERSION
from archs.instruments.synthesis import Mapping
import math
import sys

instruments_meta = Configs()
instruments_meta.read_from_file('archs/instruments/viMusic_instruments_meta.json')

class Instrument:
    def __init__(self, category='Keys', name='Grand Piano'):
        self.category = category
        self.name = name

    def get_soundfont(self):
        """Return bank id and program id in soundfont related to the sound_name"""
        try:
            program_dict = instruments_meta['instruments'][self.category][self.name]
        except:
            raise ValueError('No soundfont for {}'.format(self))
      
        return program_dict['bank'], program_dict['program']

    def __repr__(self):
        return "{} - {}".format(self.category, self.name)

class Pattern(Instrument):
    def __init__(self, category='Keys', name='Piano'):
        super().__init__(category, name)
        self.score = None

        self.ending_score = None

    def get_score(self, base_name):
        """Read midi file contain the playing pattern"""
        try:
            path = instruments_meta['patterns'][base_name][self.category][self.name]['midiPath']
        except:
            raise ValueError("Pattern's name doesn't exist [{}] {} - {}".format(base_name, self.category, self.name))

        self.score = music21.converter.parse(path)
        
        # Find any `Break` pattern in the self.category and random choose it
        break_paths = [instruments_meta['patterns'][base_name][self.category][name]['midiPath'] for name in list(instruments_meta['patterns'][base_name][self.category].keys()) if 'Break' in name]

        if len(break_paths) > 0:
            break_path = random.choice(break_paths)
            self.ending_score = music21.converter.parse(break_path)

class BaseTrack:
    def __init__(self, base_name, instrument, octave_shift = 0, patterns=None, velocity_offset=0):
        self.midi_name = ''
        self.base = base_name
        self.instrument = instrument
        self.patterns = patterns
        self.velocity_offset = velocity_offset
        self.is_drum= (self.base=='Drums')
        self.octave_shift = octave_shift

        self.get_pattern()

    def __repr__(self):
        return "Track - Type: {}, Instrument: {}. Pattern: {}".format(self.base, self.instrument, self.patterns)

    def get_soundfont(self):
        """Return bank id and program id in soundfont related to the sound_name"""
        return self.instrument.get_soundfont()

    def get_pattern(self):
        """
        Read the pattern midi files
        """
        if self.patterns is not None: #This is for `Lead` track because it doesn't have patterns
            for pattern in self.patterns:
                pattern.get_score(self.base)

    def add_track_to_score(self, input_item, in_place=True):
        """
        Add track to score
        """
        item = copy.deepcopy(input_item)
        # highest time of the input_score
        highestTime = item.highestTime

        #highest time of the backing track
        # num_time_base = math.ceil(base.highestTime)
        # num_duplicate_times = int(math.ceil(math.ceil(highestTime) / num_time_base))
        class_of_interest = ['Note', 'Chord']

        chord_symbol = item.parts[0].flat.getElementsByClass('ChordSymbol')

        ############ Check if instrument name exists
        existing_ins = [str(ins).split(':')[0] for ins in list(item.flat.getElementsByClass('Instrument'))]
        
        assign_name = ''
        type_id = 1
        
        if self.base != '':
            while True:
                temp_name = self.base + ' ' + str(type_id) 
                if not (temp_name in existing_ins):
                    assign_name = temp_name + ':' + self.instrument.name
                    break
                type_id += 1
        
        ins = music21.instrument.Instrument(instrumentName=assign_name)
        self.midi_name = assign_name
        ##############
        new_part = music21.stream.Part()
        new_part.insert(0.0, ins)
        
        if self.is_drum:
            temp_part = music21.stream.Part()

            current_number_of_bar_added = 0

            while(temp_part.highestTime < highestTime):
                random_choose_pattern_index = random.randint(0, len(self.patterns) - 1)

                current_number_of_bar_added = math.ceil(temp_part.highestTime*CONVERSION.quarterLength_to_bar_ratio())

                copied_score = copy.deepcopy(self.patterns[random_choose_pattern_index].score)

                current_offset = current_number_of_bar_added*CONVERSION.bar_to_quarterLength_ratio()
                for elem in copied_score.flat.getElementsByClass(class_of_interest):
                    temp_part.insert(elem.offset + current_offset, elem)

            elements = temp_part.getElementsByOffset(0, highestTime, includeEndBoundary=False, classList=class_of_interest)

            for elem in elements:
                new_part.insert(elem)

            #Adding break
            current_number_of_bar_added = math.ceil(new_part.highestTime*CONVERSION.quarterLength_to_bar_ratio())
            current_offset = current_number_of_bar_added*CONVERSION.bar_to_quarterLength_ratio()

            if self.patterns[random_choose_pattern_index].ending_score is not None:
                for elem in self.patterns[random_choose_pattern_index].ending_score.flat.getElementsByClass(class_of_interest):
                    new_part.insert(elem.offset + current_offset, elem)

        else:
            current_number_of_bar_added = 0

            random_choose_pattern_index = random.randint(0, len(self.patterns) - 1)

            for idx in range(len(chord_symbol)):
                start_offset = chord_symbol[idx].offset 

                #A bit complicated here, but it's running correctly
                if idx == (len(chord_symbol)-1):
                    end_offset = highestTime - start_offset
                else:
                    end_offset = chord_symbol[idx+1].offset - start_offset

                # If edit directly, it will change the reference. Need to copy
                elements = copy.deepcopy(self.patterns[random_choose_pattern_index].score.flat.getElementsByOffset(0, end_offset, includeEndBoundary=False, classList=class_of_interest))
                sys.setrecursionlimit(100000)
                for elem in elements:
                    # Try to cut the elements that last beyond the end_offset
                    if (elem.offset + elem.quarterLength) > end_offset:
                        elem.quarterLength = end_offset - elem.offset
                        
                    mapped_elem = Mapping.map_element(music21.harmony.ChordSymbol('C'),
                                                      chord_symbol[idx], elem)
                    
                    new_part.insert(start_offset + mapped_elem.offset, mapped_elem)
                    # new_part.insert(start_offset + elem.offset, elem)

                current_number_of_bar_added = math.ceil(new_part.highestTime*CONVERSION.quarterLength_to_bar_ratio())
                # Change the pattern for each 16 bar
                if ((current_number_of_bar_added % 16) == 0):
                    random_choose_pattern_index = random.randint(0, len(self.patterns) - 1)

            #Adding break
            current_number_of_bar_added = math.ceil(new_part.highestTime*CONVERSION.quarterLength_to_bar_ratio())
            current_offset = current_number_of_bar_added*CONVERSION.bar_to_quarterLength_ratio()
            
            #There's bug here, but it sounds cool :v fix it next time
            if self.patterns[random_choose_pattern_index].ending_score is not None:
                elements = copy.deepcopy(self.patterns[random_choose_pattern_index].ending_score.flat.getElementsByOffset(0, CONVERSION.bar_to_quarterLength_ratio(), includeEndBoundary=False, classList=class_of_interest))

                for elem in elements:
                    mapped_elem = Mapping.map_element(music21.harmony.ChordSymbol('C'), chord_symbol[0], elem)
                    new_part.insert(current_offset + mapped_elem.offset, mapped_elem)

            new_part.transpose(12 * self.octave_shift, inPlace=True)


        if in_place: 
            input_item.insert(new_part)        
        else:
            item.insert(new_part)        
        
        if not in_place:
            return item
        
class Lead(BaseTrack):
    def __init__(self, instrument=None, octave_shift=0, velocity_offset=0):
        super().__init__('Lead', instrument, octave_shift, None, velocity_offset)
        
    def from_score(self, score):
        #Only one part allowed
        self.score = copy.deepcopy(score)
        self.score.remove(list(self.score.flat.getElementsByClass(music21.instrument.Instrument)), recurse=True)
        self.score.transpose(12 * self.octave_shift, inPlace=True)
        
class Harmony(BaseTrack):
    def __init__(self, instrument=None, octave_shift=0, pattern=None, velocity_offset=0):
        super().__init__('Harmonics', instrument, octave_shift, pattern, velocity_offset)

class Bass(BaseTrack):
    def __init__(self, instrument=None, octave_shift=0, pattern=None, velocity_offset=0):
        super().__init__('Bass', instrument, octave_shift, pattern, velocity_offset)

class Drums(BaseTrack):
    def __init__(self, instrument=None, pattern=None, velocity_offset=0):
        super().__init__('Drums', instrument, 0, pattern, velocity_offset)