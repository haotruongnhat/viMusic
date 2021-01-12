import itertools
from common.constants import DEF, SIZE, CONVERSION

def add_dummy_tokens(itos):
    """
    The size of total vocab should be multiple of 8
    """
    if len(itos)%8 != 0:
        itos = itos + [f'dummy{i}' for i in range(len(itos)%8)]
    return itos

def vocab_list_generator(vocab):
    itos = []
    for v in vocab: 
        itos.append(v.to_list())
    itos = list(itertools.chain.from_iterable(itos))
    return add_dummy_tokens(itos)

# Definition of one event as vocab
class VocabEvent():
    """Class to handle vocab event"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __repr__(self):
        return "Vocab Event: name = '{}', prefix = '{}', size = '{}' \n".format(self.name, self.prefix, self.size)
            
    def __len__(self):
        return self.size            
    
    def __getitem__(self, idx):
        """
        Get item forward and backward
        
        If idx is number, convert number into the vocab text. For e.g: idx = 0 => {self.prefix}{idx}
        If idx is string, convert string vocab into index. For e.g: {self.prefix}{idx} => idx
        """
        if isinstance(idx, int):        
            if idx >= len(self):
                raise ValueError("{} - Index out of range. Object's length: {}".format(self.name, len(self)))
            
            indices = [i for i in range(len(self))]
            return '{}{}'.format(self.prefix, indices[idx] if len(self) > 1 else '')
        elif isinstance(idx, str):
            index = int(idx.replace(self.prefix, ''))
            return index    
        
    def to_list(self):  
        return [self[idx] for idx in range(len(self))]

class REMI:
    #Basic vocab
    BAR = VocabEvent(name='Bar', prefix='xxbar', size=1)
    PAD = VocabEvent(name='Pad', prefix='xxpad', size=1)

    POSITION = VocabEvent(name='Position', prefix='p', size=CONVERSION.bar_to_duration_ratio()) # For time sig (4/4), there're 16 position of notes, minimum is sixteenth note
    CHORD = VocabEvent(name='Chord', prefix='c', size=SIZE.FULL_CHORD)
    
    NOTE_VELOCITY = VocabEvent(name='Note Velocity', prefix='v', size=SIZE.VELOCITY) 
    NOTE_ON = VocabEvent(name='Note Pitch', prefix='n', size=SIZE.NOTE) # Midi pitch range
    NOTE_DURATION = VocabEvent(name='Note Duration', prefix='d', size=SIZE.DURATION) # Lowest duration: 1/16th note, d4 = quarter note, d32 = two white note

    # Total vocab list
    VOCAB = [BAR, PAD, POSITION, CHORD, NOTE_VELOCITY, NOTE_ON, NOTE_DURATION]
    INDEX_TOKENS = vocab_list_generator(VOCAB)
    
class MUSIC_AUTOBOT:
    BOS = VocabEvent(name='Begin Of Sequence', prefix='xxbos', size=1)
    PAD = VocabEvent(name='Pad', prefix='xxpad', size=1)
    EOS = VocabEvent(name='End Of Sequence', prefix='xxeos', size=1)
    MASK = VocabEvent(name='Mask', prefix='xxmask', size=1)
    CSEQ = VocabEvent(name='Chord Sequence', prefix='xxcseq', size=1)
    MSEQ = VocabEvent(name='Melody Sequence', prefix='xxmseq', size=1)
  
    S2SCLS = VocabEvent(name='S2SCLS', prefix='xxs2scls', size=1)
    NSCLS = VocabEvent(name='NSCLS', prefix='xxnscls', size=1)
    
    SEP = VocabEvent(name='Seperator', prefix='xxsep', size=1)
    
    NOTE = VocabEvent(name='Note', prefix='n', size=SIZE.NOTE)
    DURATION = VocabEvent(name='Duration', prefix='d', size=SIZE.DURATION)
    MTEMPO = VocabEvent(name='Tempo Changes', prefix='mt', size=10)
    
    SPECIAL_TOKS = vocab_list_generator([BOS, PAD, EOS, S2SCLS, MASK, CSEQ, MSEQ, NSCLS, SEP])
    
    VOCAB = [BOS, PAD, EOS, S2SCLS, MASK, CSEQ, MSEQ, NSCLS, SEP, NOTE, DURATION, MTEMPO]
    INDEX_TOKENS = vocab_list_generator(VOCAB)
    
    NOTE_START, NOTE_END = NOTE[0], NOTE[-1]
    DUR_START, DUR_END = DURATION[0], DURATION[-1]
