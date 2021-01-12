from common.vocab_definition import REMI, MUSIC_AUTOBOT, SIZE
from common.constants import CHORD

from common.object import Note, Chord

# Vocab - token to index mapping
class VocabItem():
    "Contain the correspondence between numbers and tokens and numericalize."
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {v:k for k,v in enumerate(self.itos)}
    
    def numericalize(self, t):
        "Convert a list of tokens `t` to their ids."
        return [self.stoi[w] for w in t]

    def textify(self, nums, sep=' '):
        "Convert a list of `nums` to their tokens."
        items = [self.itos[i] for i in nums]
        return sep.join(items) if sep is not None else items
    
    def __getstate__(self):
        return {'itos':self.itos}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.stoi = {v:k for k,v in enumerate(self.itos)}
        
    def __len__(self): 
        return len(self.itos)
    
    def save(self, path):
        "Save `self.itos` in `path`"
        pickle.dump(self.itos, open(path, 'wb'))
    
    @classmethod
    def load(cls, path):
        "Load the `Vocab` contained in `path`"
        itos = pickle.load(open(path, 'rb'))
        return cls(itos)
    
class RemiVocabItem(VocabItem):
    @property
    def bar_idx(self): return self.stoi[REMI.BAR.prefix]
    
    @property
    def pad_idx(self): return self.stoi[REMI.PAD.prefix]
    
    def chord_idx(self, pitch, quality):
        """Encode actual value into token id"""
        quality_idx = CHORD.FULL_QUALITY.index(quality) if quality in CHORD.FULL_QUALITY else  0
        return self.stoi[REMI.CHORD[pitch*len(CHORD.FULL_QUALITY) + quality_idx]]
    
    def chord_value(self, idx):
        """Decode token idx into actual value"""
        text = self.itos[idx]
        index = REMI.CHORD[text]
        
        pitch_class, quality_idx = index//len(CHORD.FULL_QUALITY), index%len(CHORD.FULL_QUALITY)
        
        return pitch_class, CHORD.FULL_QUALITY[quality_idx]
    
    @property
    def chord_idx_endpoint(self):
        """Returns position start and end index in vocab"""
        return self.stoi[REMI.CHORD[0]], self.stoi[REMI.CHORD[-1]]
    
    def position_idx(self, start):
        """Encode actual value into token id """
        
        return self.stoi[REMI.POSITION[start]] #Start position 1/16 in timesig(4/4) is 0 => pos0
    
    def position_value(self, idx):
        """Decode token idx into actual value"""
        text = self.itos[idx]
        index = REMI.POSITION[text]
    
        return index  
    
    @property
    def position_idx_endpoint(self):
        """Returns position start and end index in vocab"""
        return self.stoi[REMI.POSITION[0]], self.stoi[REMI.POSITION[-1]]
    
    def pitch_idx(self, pitch):
        return self.stoi[REMI.NOTE_ON[pitch]]
    
    def pitch_value(self, idx):
        """Decode token idx into actual value"""
        text = self.itos[idx]
        index = REMI.NOTE_ON[text]
    
        return index  
    
    @property
    def pitch_idx_endpoint(self):
        """Returns pitch start and end index in vocab 
        """
        return self.stoi[REMI.NOTE_ON[0]], self.stoi[REMI.NOTE_ON[-1]]
    
    def duration_idx(self, duration):
        """Encode actual value into token id"""
        return self.stoi[REMI.NOTE_DURATION[duration - 1]] #Duration = 1 is d0
    
    def duration_value(self, idx):
        """Decode token idx into actual value"""
        text = self.itos[idx]
        index = REMI.NOTE_DURATION[text]
    
        return index + 1
    
    @property
    def duration_idx_endpoint(self):
        """Returns duration start and end index in vocab"""
        return self.stoi[REMI.NOTE_DURATION[0]], self.stoi[REMI.NOTE_DURATION[-1]]
    
    @property
    def note_idx_range(self):
        """Contains the range object of velocity, pitch, duration"""
        return [range(self.pitch_idx_endpoint[0], self.pitch_idx_endpoint[1] + 1), 
                range(self.duration_idx_endpoint[0], self.duration_idx_endpoint[1] + 1)]
    
    @property
    def pitch_idx_range(self):
        return self.note_idx_range[0]
    
    @property
    def dur_idx_range(self):
        return self.note_idx_range[1]

    @property
    def chord_idx_range(self):
        """Contains the range object of chord symbol"""
        return [range(self.chord_idx_endpoint[0], self.chord_idx_endpoint[1] + 1)]
    
    @property
    def position_idx_range(self):
        return [range(self.position_idx_endpoint[0], self.position_idx_endpoint[1] +1)]

    def note_to_tokens(self, note:Note):
        return [self.position_idx(note.start), self.pitch_idx(note.pitch), self.duration_idx(note.duration % SIZE.DURATION)]
        
    def tokens_to_note(self, tokens):
        pos_value = self.position_value(tokens[0])
        pitch = self.pitch_value(tokens[1])
        duration = self.duration_value(tokens[2])
   
        return Note(start=pos_value, duration=duration, pitch=pitch, velocity=100)
        
    def chord_to_tokens(self, chord:Chord):      
        return [self.position_idx(chord.start), self.chord_idx(chord.pitch, chord.quality)]
   
    def tokens_to_chord(self, tokens):
        pos_value = self.position_value(tokens[0])
        pitch, quality = self.chord_value(tokens[1])
        return Chord(start=pos_value, pitch=pitch, quality=quality)
   
class RemiMidiVocabItem(RemiVocabItem):
    """This class to handle velocity midi event"""
    def velocity_idx(self, velocity):
        """Encode actual value into token id"""
        return self.stoi[REMI.NOTE_VELOCITY[velocity]] #Velocity = 0 => vel0
    
    def velocity_value(self, idx):
        """Decode token idx into actual value"""
        text = self.itos[idx]
        index = REMI.NOTE_VELOCITY[text]
    
        return index  
    
    @property
    def velocity_idx_endpoint(self):
        """Returns velocity start and end index in vocab"""
        return self.stoi[REMI.NOTE_VELOCITY[0]], self.stoi[REMI.NOTE_VELOCITY[-1]]
     
    @property
    def note_idx_range(self):
        """Contains the range object of velocity, pitch, duration"""
        return [range(self.velocity_idx_endpoint[0], self.velocity_idx_endpoint[1] + 1), 
                range(self.pitch_idx_endpoint[0], self.pitch_idx_endpoint[1] + 1), 
                range(self.duration_idx_endpoint[0], self.duration_idx_endpoint[1] + 1)]

    def note_to_tokens(self, note:Note):
        return [self.position_idx(note.start), self.velocity_idx(note.velocity), self.pitch_idx(note.pitch), self.duration_idx(note.duration)]

    def tokens_to_note(self, tokens):
        pos_value = self.position_value(tokens[0])
        velocity = self.velocity_value(tokens[1])
        pitch = self.pitch_value(tokens[2])
        duration = self.duration_value(tokens[3])
   
        return Note(start=pos_value, duration=duration, pitch=pitch, velocity=velocity)

class MusicAutobotVocabItem(VocabItem):
    @property 
    def mask_idx(self): return self.stoi[MUSIC_AUTOBOT.MASK.prefix]
    @property 
    def pad_idx(self): return self.stoi[MUSIC_AUTOBOT.PAD.prefix]
    @property
    def bos_idx(self): return self.stoi[MUSIC_AUTOBOT.BOS.prefix]
    @property
    def sep_idx(self): return self.stoi[MUSIC_AUTOBOT.SEP.prefix]
    @property
    def npenc_range(self): return (self.stoi[MUSIC_AUTOBOT.SEP.prefix], self.stoi[MUSIC_AUTOBOT.DUR_END]+1)
    @property
    def note_range(self): return self.stoi[MUSIC_AUTOBOT.NOTE_START], self.stoi[MUSIC_AUTOBOT.NOTE_END]+1
    @property
    def dur_range(self): return self.stoi[MUSIC_AUTOBOT.DUR_START], self.stoi[MUSIC_AUTOBOT.DUR_END]+1

    def is_duration(self, idx): 
        return idx >= self.dur_range[0] and idx < self.dur_range[1]
    def is_duration_or_pad(self, idx):
        return idx == self.pad_idx or self.is_duration(idx)

