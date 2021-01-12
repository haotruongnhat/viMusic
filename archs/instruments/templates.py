from archs.instruments.track import Instrument, Pattern, Lead, Harmony, Bass, Drums
from archs.instruments.synthesis import Synthesis
import music21
import copy
from common.object import Configs
import time
from common.constants import CONVERSION
import random
import numpy as np

class Template():
    @staticmethod
    def create_from_config(config_dict):
        return Template(
            leads = [Lead(Instrument(**config["instrument"]), config["octave_shift"], config["velocity_offset"]) for config in config_dict["leads"]],
            harmonics=[Harmony(Instrument(**config["instrument"]), config["octave_shift"], [Pattern(**conf) for conf in config["pattern"]], config["velocity_offset"]) for config in config_dict["harmonics"]],
            basses=[Bass(Instrument(**config["instrument"]), config["octave_shift"], [Pattern(**conf) for conf in config["pattern"]], config["velocity_offset"]) for config in config_dict["basses"]],
            drums=[Drums(Instrument(**config["instrument"]), [Pattern(**conf) for conf in config["pattern"]], config["velocity_offset"]) for config in config_dict["drums"]],
        )
    
    def __init__(self, leads, harmonics=[], basses=[], drums=[]):
        self.leads = leads
        self.harmonics = harmonics
        self.basses = basses
        self.drums = drums
        
    @property
    def num_tracks(self):
        return len(self.leads + self.harmonics + self.basses + self.drums)

    def assign_lead_melody(self, score, index=-1):
        """
        Assign melody into lead tracks. If index == -1: pass melody into all the lead tracks
        """
        if index >= len(self.leads):
            raise ValueError('Index not valid for leads')
        
        if (len(score.flat.getElementsByClass('ChordSymbol')) == 0):
            raise ValueError('No chord symbol in this score')

        if (len(score.parts) == 0):
            raise ValueError('Theres {} parts in this score'.format(len(score.parts)))

        if index == -1:
            for idx in range(len(self.leads)):
                self.assign_lead_melody(score, idx)
        else:
            self.leads[index].from_score(score)
        
    def to_score(self):
        """Merge all the tracks into one score"""
        # Merge the leads first
        score = music21.stream.Score()
        
        print('to_score:: Insert Lead track')
        for lead in self.leads:
            for idx in range(len(lead.score.parts)):
                part = copy.deepcopy(lead.score.parts[idx])
                ins = Synthesis.create_track_instrument(score, lead.base, lead.instrument.name)
                part.insert(0.0, ins)
                
                score.insert(part)
            lead.midi_name = str(ins)
    
        # Then merge the backing track (Harmonics, Basses, Drums)
        for idx, backing_track in enumerate(self.harmonics + self.basses + self.drums):
            print('to_score:: Insert {} track'.format(backing_track.base))

            backing_track.add_track_to_score(score)
            # Synthesis.add_backing_track(score, backing_track.score, 
            #                             backing_track.base, 
            #                             backing_track.instrument.name,
            #                             backing_track.octave_shift,
            #                             backing_track.is_drum,
            #                             in_place=True)
            
        # instrument_list = score.flat.getElementsByClass('Instrument')
        # for idx, track in enumerate(self.leads + self.harmonics + self.basses + self.drums):
        #     track.midi_name = str(instrument_list[idx])
    
        return score

    def to_dynamic_instrument_score(self, changing_each_measure = 16):
        score = self.to_score()
        quarterLength_step = CONVERSION.bar_to_quarterLength_ratio()*changing_each_measure

        # Find start index of each type of track
        # For e.g; We have 2 lead track, 3 harmonic track, 2 bass track, 2 drums track
        # The track start index will be: 0 - 2 - 5 - 7 - 9
        type_track_start_index = [0] + list(np.cumsum([len(self.leads), len(self.harmonics), len(self.basses), len(self.drums)])[:-1])

        for type_index, type_track in enumerate([self.leads, self.harmonics, self.basses, self.drums]):
            target = type_track
            if len(target) > 1:
                step_idx = 0
                if type_index == 3: #Drums has 16 bar per pattern
                    quarterLength_step = CONVERSION.bar_to_quarterLength_ratio()*16
                    
                base_index = list(range(0, len(target)))

                while (quarterLength_step * step_idx) < score.flat.highestOffset:
                    #Only remove instrument on even step_idx
                    if (step_idx%2) == 0:
                        if len(base_index) == 0:
                            base_index = list(range(0, len(target)))
                        
                        #Only one drum and bass track allowed to play each time
                        if type_index in [2, 3]:
                            base_index = list(range(0, len(target)))
                            base_index.pop(base_index.index(random.choice(base_index)))
                            
                            for idx in base_index:#base_index:
                                part_in_score = type_track_start_index[type_index] + idx
                                removed_elements = list(score.parts[part_in_score].flat.getElementsByOffset(quarterLength_step * step_idx, quarterLength_step * (step_idx+1), includeEndBoundary=False, classList=['Note', 'Chord']))
                                score.remove(removed_elements, recurse=True)
                        elif type_index in [0]: ## Ignore leads in this removing step (because want to play instrument and humming along)
                            pass
                        else:
                            chosen_index = base_index.pop(base_index.index(random.choice(base_index)))
                            
                            for idx in [chosen_index]:#base_index:
                                part_in_score = type_track_start_index[type_index] + idx
                                removed_elements = list(score.parts[part_in_score].flat.getElementsByOffset(quarterLength_step * step_idx, quarterLength_step * (step_idx+1), includeEndBoundary=False, classList=['Note', 'Chord']))
                                score.remove(removed_elements, recurse=True)
                        
                    step_idx += 1
                        
        return score
        
    def to_midi(self, tempo=120, dynamical=False):
        score = copy.deepcopy(self.to_score() if not dynamical else self.to_dynamic_instrument_score())
        score.remove(list(score.flat.getElementsByClass(music21.harmony.ChordSymbol)), recurse=True)        

        midi = Synthesis.convert_score_to_pretty_mid(score, tempo)

        tracks = self.leads + self.harmonics + self.basses + self.drums
        track_midi_name = [track.midi_name for track in tracks]

        # Changing the program_change (bank id and program id)
        for ins in midi.instruments:    
            index = track_midi_name.index(ins.name)
            
            bank_id, program_id = tracks[index].get_soundfont()
            
            ins.bank = bank_id
            ins.program = program_id
            ins.is_drum= tracks[index].is_drum
            ins.amplitude_offset = tracks[index].velocity_offset

        print('Intrument synthesis')

        return midi

    def to_midi_file(self, fp, tempo=120, dynamical=False):
        self.to_midi(tempo, dynamical).write(fp)

    def to_wav_array(self, tempo=120, dynamical=False):
        return Synthesis.to_wav_array(self.to_midi(tempo, dynamical))

    def to_wav_file(self, fp, tempo=120, dynamical=False):
        return Synthesis.to_wav_file(self.to_midi(tempo,dynamical),fp)
        
    def play(self, tempo=120, dynamical=False):
        return Synthesis.play_midi(self.to_midi(tempo, dynamical))