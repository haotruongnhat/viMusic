import copy
import json
from common.constants import CHORD, CONVERSION
import music21
from g2p_en import G2p

class Dict(dict):
    """
        Dictionary processing method
    """
    def read_from_file(self, fp):
        with open(fp, 'r') as f:
            self.update(json.load(f))

    def write_to_file(self, fp):
        with open(fp, 'w') as f:
            json.dump(self, f)
            
    def __getattr__(self, key):
        return self[key]
    
    def __setattr__(self, key, value):
        self[key] = value

class Configs(Dict):
    """Class to handle configs"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self[key] = value
            setattr(self, key, value)

    def merge(self, other_dict, in_place=False):
        """Merge two config"""
        assert isinstance(other_dict, dict)
        og_dict = copy.deepcopy(self)
        og_dict.update(other_dict)

        if not in_place:
            return og_dict
        else:
            self.__init__(**og_dict)

    def __add__(self, other_dict):
        assert isinstance(other_dict, dict)
        return self.merge(other_dict, in_place=False)

class HParams(Configs):
    """Class to handle parameters"""
    pass

class Note(object):
    def __init__(self, start=None, duration=None, pitch=None, velocity=None, syllable='',syll_emb=None):
        """
        Initialize the Note class
        
        Args:

            start (int): Absolute start time in duration (not in quarterLength)
            duration (int): Duration time in duration (not in quarterLength)
            pitch (int): MIDI pitch
            velocity (int): How hard the note being hit
            syllable (str): Syllable attached into the note
        """
        self.priority = 1
        self.start = start
        self.duration = duration
        
        self.pitch = pitch
        self.velocity = velocity
        self.syllable = syllable
        self.syll_emb = syll_emb
        
    def __repr__(self):
        return "Note - start: {}, duration: {}, pitch: {}, velocity: {}, syllable: {}, vector {} \n".format(self.start, self.duration, self.pitch, self.velocity, self.syllable,self.syll_emb)    
    
    def start_time_from_measure(self, measure_nth, 
                                time:music21.meter.TimeSignature = music21.meter.TimeSignature('4/4')):
        pass
    
    def at_measure(self, time:music21.meter.TimeSignature = music21.meter.TimeSignature('4/4')):
        """Calculate measure id the event is standing on"""
        return int(CONVERSION.duration_to_bar_ratio(time) * self.start)
    
    def to_absolute_note(self, measure_id, time:music21.meter.TimeSignature = music21.meter.TimeSignature('4/4'), in_place=False):
        """
        Turn object into absolute start time to the beginning of score
        
        Args:
        
            time (music21 TimeSignature): Time Signature of the score
            in_place (bool): Replace the inside parameters or return the new one
        
        Returns:

            Note: absolute start time note (if in_place=False)
        """
        abs_start = measure_id*CONVERSION.bar_to_duration_ratio(time) + self.start
        
        if in_place == True:
            self.start = abs_start
        else:
            return Note(abs_start, self.duration, self.pitch, self.velocity, self.syllable,self.syll_emb)
    
    def to_relative_note_to_bar(self, time:music21.meter.TimeSignature = music21.meter.TimeSignature('4/4'), in_place=False):
        """
        Turn object into relative start time to closed bar beginning
        
        Args:
        
            time (music21 TimeSignature): Time Signature of the score
            in_place (bool): Replace the inside parameters or return the new one
        
        Returns:

            Note: relative start time note (if in_place=False)
        """
        
        measure_id = self.at_measure(time)
        relative_start = self.start - measure_id * CONVERSION.bar_to_duration_ratio(time)
    
        if in_place == True:
            self.start = relative_start
        else:
            return Note(relative_start, self.duration, self.pitch, self.velocity, self.syllable,self.syll_emb)
    
    def from_music21_note(self, note):
        """Parse from music21's note"""
        self.start = CONVERSION.quarterLength_to_duration(note.offset)
        self.duration = CONVERSION.quarterLength_to_duration(note.quarterLength)
        
        self.pitch = int(note.pitch.ps)
        self.velocity = note.volume.velocity if note.volume.velocity else 90
        self.syllable = note.lyric

        self.syllable = note.lyric #?

    def from_pretty_midi_note(self, note):
        """Parse from music21's note"""
        self.start = CONVERSION.quarterLength_to_duration(note.start)
        self.duration = CONVERSION.quarterLength_to_duration(note.end-note.start)
        
        self.pitch = int(note.pitch)
        try:
            self.velocity = note.velocity     
        except:
            self.velocity = 100

    def to_music21_note(self):
        """Return music21's note"""
        note = music21.note.Note(self.pitch)
        note.quarterLength = CONVERSION.duration_to_quarterLength(self.duration)
        note.offset = CONVERSION.duration_to_quarterLength(self.start)
        note.lyric = self.syllable

        return note
            
class Chord(object):
    def __init__(self, start=None, pitch=None, quality=None, duration=None):
        """
        Class intialization.
        
        Args:
        
            start (int): Absolute start time in duration
            pitch (int): MIDI pitch
            quality(str): Chord's quality. See definition in CHORD.FULL_QUALITY
        """
        self.priority = 0 # This is for sorting
        self.start = start
        self.pitch = pitch
        self.quality = quality
        self.duration = duration
        
    def __repr__(self):
        return "Chord - start: {}, pitch class: {} ('{}'), quality: {}\n".format(self.start, self.pitch, CHORD.PITCH_NAME[self.pitch], self.quality)    
    
    def from_music21_chord(self, chord):
        """ Parse music21's chord"""
        
        self.start = CONVERSION.quarterLength_to_duration(chord.offset)
        self.pitch = chord[0].pitch.pitchClass
        self.quality = chord.quality
        self.duration = CONVERSION.quarterLength_to_duration(chord.quarterLength)

    def from_miditoolkit_chord(self, chord):

        self.start = chord[0]
        self.pitch = music21.note.Note(chord[-3]).pitch.pitchClass

        if chord[-2]=='maj': self.quality='major'
        elif chord[-2]=='min': self.quality='minor'
        elif chord[-2]=='dim': self.quality='diminished'
        else: self.quality='other'

        self.duration = chord[1]-chord[0]

    def to_music21_chord(self, playable=True):
        """Return music21's chord"""
        
        ### Remove tie on too long chord (>1 measure)
        start_measure = self.at_measure()
        end_measure = self.end_at_measure()
        
        chords = []
        
        start_time = self.start
        for measure_id in range(start_measure, end_measure + 1):
            # From start_time to closed bar_end
            end_time = (measure_id+1)*CONVERSION.bar_to_duration_ratio()
            if end_time > (self.start + self.duration): end_time = (self.start + self.duration) 
            
            chord = Chord(start = start_time, pitch= self.pitch, quality=self.quality, duration= end_time - start_time)
            chords.append(chord)
            start_time = end_time
        
        def chord_to_m21_chord(chord, playable=True):
            """Convert `Chord` into `music21::ChordSymbol"""
            
            m21_chord = music21.harmony.ChordSymbol(root=music21.pitch.Pitch(chord.pitch), kind = chord.quality)
            if playable:
                m21_chord.writeAsChord = True
                m21_chord.quarterLength = CONVERSION.duration_to_quarterLength(chord.duration)
            m21_chord.offset = CONVERSION.duration_to_quarterLength(chord.start)
            
            return m21_chord
        
        if len(chords) == 1:
            return chord_to_m21_chord(chords[0], playable)
        else:
            return [chord_to_m21_chord(c, playable) for c in chords]

    def end_at_measure(self, time:music21.meter.TimeSignature = music21.meter.TimeSignature('4/4')):
        return int(CONVERSION.duration_to_bar_ratio(time) * (self.start + self.duration - 0.1))


    def at_measure(self, time:music21.meter.TimeSignature = music21.meter.TimeSignature('4/4')):
        """
        Find the measure index where the current object is standing (0-based)
        
        Args:
        
            time (music21 TimeSignature): Time Signature of the score
        """
        return int(CONVERSION.duration_to_bar_ratio(time) * self.start)
    
    def to_relative_note_to_bar(self, time:music21.meter.TimeSignature = music21.meter.TimeSignature('4/4'), in_place=False):
        """
        Turn object into relative start time to closed bar beginning
        
        Args:
        
            time (music21 TimeSignature): Time Signature of the score
            in_place (bool): Replace the inside parameters or return the new one
        
        Returns:

            Chord: relative start time chord (if in_place=False)
        """
        
        measure_id = self.at_measure(time)
        relative_start = self.start - measure_id * CONVERSION.bar_to_duration_ratio(time)
    
        if in_place == True:
            self.start = relative_start
        else:
            return Chord(relative_start, self.pitch, self.quality, self.duration)

    def to_absolute_chord(self, measure_id, time:music21.meter.TimeSignature = music21.meter.TimeSignature('4/4'), in_place=False):
        """Turn object into absolute start time relative to beginning of score
        
        Args:
        
            time (music21 TimeSignature): Time Signature of the score
            in_place (bool): Replace the inside parameters or return the new one
        
        Returns:

            Chord: absolute start time chord (if in_place=False)
        """
        
        abs_start = measure_id*CONVERSION.bar_to_duration_ratio(time) + self.start
        
        if in_place == True:
            self.start = abs_start
        else:
            return Chord(abs_start, self.pitch, self.quality, self.duration)

class G2PSingleton:
    class __G2PSingleton:
        def __init__(self):
            self.g2p = G2p()

        def __call__(self, text):
            return self.g2p(text)

    _instance = None

    @classmethod
    def init(cls):
        if not cls._instance:
            cls._instance = cls.__G2PSingleton()

    @classmethod
    def pad(cls):
        return cls.itos()[0]

    @classmethod
    def unknown(cls):
        return cls.itos()[1]

    @classmethod
    def start(cls):
        return cls.itos()[2]

    @classmethod
    def end(cls):
        return cls.itos()[3]

    @classmethod
    def to_phonemes(cls, text):
        cls.init()
        return cls._instance(text)

    @classmethod
    def stoi(cls):
        cls.init()
        return cls._instance.g2p.p2idx
    
    @classmethod
    def itos(cls):
        cls.init()
        return cls._instance.g2p.phonemes