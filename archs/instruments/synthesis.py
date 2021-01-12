import sys
import music21
from pathlib import Path
from IPython.display import Audio
from common.music_item import *
from common.pretty_midi.pretty_midi import PrettyMIDI
from common.pretty_midi.instrument import Instrument
from common.pretty_midi.utilities import instrument_name_to_program
import copy
import numpy as np
import math


class Mapping:
    """ This class functions is to mapping between Key and PitchClass 
        E.g: Want to map note(C#) from chord(C-major) to chord(E-minor)
    """

    @staticmethod
    def chord_to_map(chord):
        return [p.pitchClass for p in music21.key.Key(chord).pitches]

    @staticmethod
    def map_pitch(source_map, target_map, pitch):
        mapped_pitchClass = target_map[source_map.index(pitch.pitchClass if pitch.pitchClass in source_map else pitch.transpose(-1).pitchClass)]
        
        return music21.pitch.Pitch(mapped_pitchClass)

    @staticmethod
    def map_chord(source_chord, target_chord, chord):
        source_map = Mapping.chord_to_map(source_chord)
        target_map = Mapping.chord_to_map(target_chord)
        
        new_pitches = [Mapping.map_pitch(source_map, target_map, pitch) for pitch in chord.pitches]
        for p in new_pitches:
            p.octave = chord.root().octave
        
        mapped_chord = music21.chord.Chord(new_pitches)
        mapped_chord.offset = chord.offset
        mapped_chord.quarterLength = chord.quarterLength
        
        return mapped_chord

    @staticmethod
    def map_note(source_chord, target_chord, note):
        source_map = Mapping.chord_to_map(source_chord)
        target_map = Mapping.chord_to_map(target_chord)

        pitch = note.pitch
        mapped_note = music21.note.Note(Mapping.map_pitch(source_map, target_map, pitch))
        mapped_note.offset = note.offset
        mapped_note.quarterLength = note.quarterLength
        mapped_note.octave = note.octave
        
        return mapped_note

    @staticmethod
    def map_element(source_chord, target_chord, elem):
        if isinstance(elem, music21.chord.Chord):
            return Mapping.map_chord(source_chord, target_chord, elem)
        elif isinstance(elem, music21.note.Note):
            return Mapping.map_note(source_chord, target_chord, elem)
        else:
            raise TypeError('Not valid type to do mapping')

class Synthesis:
    DEFAULT_SOUNDFONT_PATH = 'archs/instruments/viMusic.sf2'
    DEFAULT_SAMPLE_RATE = 44100
    
    @staticmethod
    def convert_score_to_pretty_mid(score, tempo=120, resolution = 1024):

        # import inspect
        # curframe = inspect.currentframe()
        # calframe = inspect.getouterframes(curframe,2)
        # print('caller name:',calframe[1][3])

        pretty = PrettyMIDI(resolution=resolution, initial_tempo=tempo)#score.flat.getElementsByClass('MetronomeMark')[0].number)
        for part in score.parts:
            try:
                ins = Instrument(program = 0, name = part.flat.getElementsByClass('Instrument')[0].instrumentName)
            except:
                ins = Instrument(program = 0, name = 'Acoustic Grand Piano')

            def music21_note_to_note(m21_note, offset = None):
                n = Note()
                n.from_music21_note(m21_note)
                
                if offset is not None:
                    n.start = CONVERSION.quarterLength_to_duration(offset)
                    
                return n
                
            sequence = []
            elems = part.flat.getElementsByClass(['Note', 'Chord'])
            for elem in elems:
                if isinstance(elem, music21.note.Note):
                    sequence.append(music21_note_to_note(elem))
                else:
                    for n in elem.notes:
                        sequence.append(music21_note_to_note(n, offset = elem.offset))
            
            for note in sequence:
                start = note.start
                end = note.start + note.duration
                start = CONVERSION.duration_to_quarterLength(start)
                end = CONVERSION.duration_to_quarterLength(end)
                start *= resolution
                end *= resolution
                start = pretty.tick_to_time(int(start))
                end = pretty.tick_to_time(int(end))
                
                n = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch,start=start,end=end)
                ins.notes.append(n)
            
            pretty.instruments.append(ins)
        
        return pretty

    @staticmethod
    def to_wav_array(midi_obj, sample_rate = DEFAULT_SAMPLE_RATE, soundfont_path = DEFAULT_SOUNDFONT_PATH):
        return midi_obj.fluidsynth(fs= sample_rate, sf2_path = soundfont_path)

    @staticmethod
    def to_wav_file(midi_obj, output_file=None, sample_rate = DEFAULT_SAMPLE_RATE, soundfont_path = DEFAULT_SOUNDFONT_PATH):
        return midi_obj.write_wav(output_file, sample_rate, soundfont_path)
        
    @staticmethod
    def play_midi(midi_obj, sample_rate = DEFAULT_SAMPLE_RATE, soundfont_path = DEFAULT_SOUNDFONT_PATH):
        return Audio(Synthesis.to_wav_array(midi_obj, sample_rate, soundfont_path), rate= sample_rate)

    @staticmethod
    def create_track_instrument(score, base_name='', instrument_name=''):
        existing_ins = [str(ins).split(':')[0] for ins in list(score.flat.getElementsByClass('Instrument'))]
        assign_name = ''
        type_id = 1
        
        if base_name != '':
            while True:
                temp_name = base_name + ' ' + str(type_id) 
                if not (temp_name in existing_ins):
                    assign_name = temp_name + ':' + instrument_name
                    break
                type_id += 1
        
        ins = music21.instrument.Instrument(instrumentName=assign_name)
        return ins
    
    @staticmethod
    def add_backing_track(input_item, input_base, part_name='', instrument_name='', octave_shift = 0, is_drum=False, in_place=False):
        item = copy.deepcopy(input_item)
        base = copy.deepcopy(input_base)
        
        highestTime = item.highestTime
        num_time_base = math.ceil(base.highestTime)
        num_duplicate_times = int(math.ceil(math.ceil(highestTime) / num_time_base))
        class_of_interest = ['Note', 'Chord']
        
        chord_symbol = item.parts[0].flat.getElementsByClass('ChordSymbol')
        
        ############ Check if instrument name exists
        existing_ins = [str(ins).split(':')[0] for ins in list(item.flat.getElementsByClass('Instrument'))]
        
        assign_name = ''
        type_id = 1
        
        if part_name != '':
            while True:
                temp_name = part_name + ' ' + str(type_id) 
                if not (temp_name in existing_ins):
                    assign_name = temp_name + ':' + instrument_name
                    break
                type_id += 1
        
        ins = music21.instrument.Instrument(instrumentName=assign_name)
        ##############
        
        new_part = music21.stream.Part()
        new_part.insert(0.0, ins)
        
        if is_drum:
            for elem in base.flat.getElementsByClass(class_of_interest):
                base.parts[0].repeatInsert(elem, [elem.offset + start_time_measure for start_time_measure in range(num_time_base, num_duplicate_times*num_time_base, num_time_base)])
        
            elements = base.flat.getElementsByOffset(0, highestTime, includeEndBoundary=False, classList=class_of_interest)
            for elem in elements:
                new_part.insert(elem)
        else:
            for idx in range(len(chord_symbol)):
                start_offset = chord_symbol[idx].offset 
                
                #A bit complicated here, but it's running correctly
                if idx == (len(chord_symbol)-1):
                    end_offset = highestTime - start_offset
                else:
                    end_offset = chord_symbol[idx+1].offset - start_offset

                # If edit directly, it will change the reference. Need to copy
                elements = copy.deepcopy(base.flat.getElementsByOffset(0, end_offset, includeEndBoundary=False, classList=class_of_interest))
                
                # for elem in elements.transpose(chord_symbol[idx].root().pitchClass):
                for elem in elements:
                    # Try to cut the elements that beyond the end_offset
                    if (elem.offset + elem.quarterLength) > end_offset:
                        elem.quarterLength = end_offset - elem.offset
                        
                    mapped_elem = Mapping.map_element(music21.harmony.ChordSymbol('C'),
                                                      chord_symbol[idx], elem)
                    
                    new_part.insert(start_offset + mapped_elem.offset, mapped_elem)
                    # new_part.insert(start_offset + elem.offset, elem)
                    
            new_part.transpose(12*octave_shift, inPlace=True)

        if in_place: 
            input_item.insert(new_part)        
        else:
            item.insert(new_part)        
        
        if not in_place:
            return item