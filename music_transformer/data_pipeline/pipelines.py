import magenta
from magenta.music import sequences_lib
from magenta.music import chord_symbols_lib
from magenta.music.constants import *
from music_transformer.constants import *

from music_transformer.music.protobuf import vimusic_pb2
from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
import tensorflow as tf
import copy, math

from .data_lib import ViMusic
from .data_lib import unify_beat_in_sequence

from fractions import Fraction

CHORD_SYMBOL = vimusic_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


class NegativeTimeError(Exception):
    """
    Raised when end_time is smaller than start_time
    """
    pass


class ViMusicWithLyricsExtractor(pipeline.Pipeline):
    """Extracts polyphonic tracks and lyrics from a quantized NoteSequence."""
    
    def __init__(self, min_events, max_events, num_velocity_bins, name=None):
        super(ViMusicWithLyricsExtractor, self).__init__(
        input_type=vimusic_pb2.NoteSequence,
        output_type=ViMusic,
        name=name)
        self._min_events = min_events
        self._max_events = max_events
        self._num_velocity_bins = num_velocity_bins

    def transform(self, sequence):
        """It's extraction time :>"""

        # pylint: disable=g-complex-comprehension
        #Getting stats
        stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
        ['performances_truncated',
         'performances_discarded'])

        #always allow more than 1 program
        instruments = set(note.instrument for note in sequence.notes)

        performances = []

        for instrument in instruments:
            # Translate the sequence into self-made performance.
            performance = ViMusic(sequence, start_step=0,
                                num_velocity_bins=self._num_velocity_bins,
                                instrument=instrument)
            if (self._max_events is not None and
            len(performance) > self._max_events):
                performance.truncate(self._max_events)
                stats['performances_truncated'].increment()

            if self._min_events is not None and len(performance) < self._min_events:
                stats['performances_discarded'].increment()
                
            performances.append(performance)

        return performances

class ViSustainPipeline(pipeline.Pipeline):

    """Applies sustain pedal control changes to a NoteSequence."""
    def __init__(self,name=None):
        super(ViSustainPipeline, self).__init__(
        input_type=vimusic_pb2.NoteSequence,
        output_type=vimusic_pb2.NoteSequence,
        name=name)
  

    def transform(self, note_sequence):
        return [sequences_lib.apply_sustain_control_changes(note_sequence)]

class ViSplitterPipeline(pipeline.Pipeline):
    """A Pipeline that splits NoteSequences at regular intervals."""

    def __init__(self, hop_size_seconds, name=None):
        """Creates a Splitter pipeline.
        Args:
        hop_size_seconds: Hop size in seconds that will be used to split a
            NoteSequence at regular intervals.
        name: Pipeline name.
        """
        super(ViSplitterPipeline, self).__init__(
        input_type=vimusic_pb2.NoteSequence,
        output_type=vimusic_pb2.NoteSequence,
        name=name)
        self._hop_size_seconds = hop_size_seconds

    def transform(self, note_sequence):
        #do some simple modification to the sequence_libs
        return sequences_lib.split_note_sequence(
        note_sequence, self._hop_size_seconds)
        

class ConcatenateLyricsPipeline(pipeline.Pipeline):
    def __init__(self,name=None):
        super(ConcatenateLyricsPipeline, self).__init__(
        input_type=vimusic_pb2.NoteSequence,
        output_type=vimusic_pb2.NoteSequence,
        name=name)

    def transform(self, quantized_sequence):
        """
        Perform concanetation operation to the Note Sequence
        transform all notes holding text that has syllabic of "end", "middle", "single" to "single"
        """
        copy_quantized_sequence = copy.deepcopy(quantized_sequence)
        notes = [note for note in copy_quantized_sequence.notes]
        sorted_notes = sorted(notes, key=lambda note: (note.start_time, note.pitch))
        syllable_bag = list()
        for note in sorted_notes:
            if note.syllabic == "begin" or note.syllabic == "middle":
                syllable_bag.append(note)
            if note.syllabic == "end":
                syllable_bag.append(note)
                #begin concantenate to word
                word = ''.join([n.text for n in syllable_bag])
                for n in syllable_bag:
                    n.text = word
                syllable_bag = list()
        
        return [copy_quantized_sequence]

class UnifyNotesPipeline(pipeline.Pipeline):
    def __init__(self,base_pitch,name=None,instrument=1):
        super(UnifyNotesPipeline, self).__init__(
        input_type=vimusic_pb2.NoteSequence,
        output_type=vimusic_pb2.NoteSequence,
        name=name)
        self._base_pitch = base_pitch
        self._instrument = instrument


    def transform(self,sequence):
        unify_sequence = copy.deepcopy(sequence)
        chord_symbols = [x for x in unify_sequence.text_annotations 
        if x.annotation_type == CHORD_SYMBOL if x.time <= sequence.total_time]
        if len(chord_symbols) > 0:
            #special case for last chord_symbol
            infer_notes = chord_symbols_lib.chord_symbol_pitches(chord_symbols[-1].text)
            infer_pitches = [x + self._base_pitch for x in infer_notes]
            start_time = chord_symbols[-1].time
            end_time = unify_sequence.total_time
            for p in infer_pitches:
                note = unify_sequence.notes.add()
                note.pitch = p
                note.instrument = self._instrument
                note.start_time = start_time
                note.end_time = end_time

            for idx, ta in enumerate(chord_symbols[:-1][::-1]):
                infer_notes = chord_symbols_lib.chord_symbol_pitches(ta.text)
                infer_pitches = [x + self._base_pitch for x in infer_notes]
                start_time = ta.time
                end_time = chord_symbols[len(chord_symbols) - idx - 1].time
                for p in infer_pitches:
                    note = unify_sequence.notes.add()
                    note.pitch = p
                    note.instrument = self._instrument
                    note.start_time = start_time
                    note.end_time = end_time
                    

        #delete chord symbol
        i = 0
        while i < len(unify_sequence.text_annotations):
            if unify_sequence.text_annotations[i].annotation_type != CHORD_SYMBOL:
                i += 1
            else:
                del unify_sequence.text_annotations[i]

        #Get list of tempos
        #Using an implicit bpm of 120.0
        unify_sequence = unify_beat_in_sequence(unify_sequence,DEFAULT_BPM)
        return [unify_sequence]

class ViQuantizerPipeline(pipeline.Pipeline):
    """A Pipeline that quantizes NoteSequence data."""

    def __init__(self, steps_per_quarter=None, steps_per_second=None, name=None):
        """Creates a Quantizer pipeline.

        Exactly one of `steps_per_quarter` and `steps_per_second` should be defined.

        Args:
        steps_per_quarter: Steps per quarter note to use for quantization.
        steps_per_second: Steps per second to use for quantization.
        name: Pipeline name.

        Raises:
        ValueError: If both or neither of `steps_per_quarter` and
            `steps_per_second` are set.
        """
        super(ViQuantizerPipeline, self).__init__(
        input_type=vimusic_pb2.NoteSequence,
        output_type=vimusic_pb2.NoteSequence,
        name=name)
        if (steps_per_quarter is not None) == (steps_per_second is not None):
            raise ValueError(
            'Exactly one of steps_per_quarter or steps_per_second must be set.')
        self._steps_per_quarter = steps_per_quarter
        self._steps_per_second = steps_per_second

    def transform(self, note_sequence):
        try:
            if self._steps_per_quarter is not None:
                quantized_sequence = sequences_lib.quantize_note_sequence(
                note_sequence, self._steps_per_quarter)
            else:
                quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
                note_sequence, self._steps_per_second)
            return [quantized_sequence]
        except sequences_lib.MultipleTimeSignatureError as e:
            tf.logging.warning('Multiple time signatures in NoteSequence %s: %s',
                         note_sequence.filename, e)
            self._set_stats([statistics.Counter(
            'sequences_discarded_because_multiple_time_signatures', 1)])
            return []
        except sequences_lib.MultipleTempoError as e:
            tf.logging.warning('Multiple tempos found in NoteSequence %s: %s',
                         note_sequence.filename, e)
            self._set_stats([statistics.Counter(
            'sequences_discarded_because_multiple_tempos', 1)])
            return []
        except sequences_lib.BadTimeSignatureError as e:
            tf.logging.warning('Bad time signature in NoteSequence %s: %s',
                         note_sequence.filename, e)
            self._set_stats([statistics.Counter(
            'sequences_discarded_because_bad_time_signature', 1)])
            return []

 
class ViMusicEncoderPipeline(pipeline.Pipeline):
    def __init__(self,config,name=None):
        super(ViMusicEncoderPipeline, self).__init__(
        input_type=ViMusic,
        output_type=tf.train.SequenceExample,
        name=name)

        self._encoder_decoder = config.encoder_decoder

    def transform(self,vi_events):
        encoded = [self._encoder_decoder.encode(vi_events)]
        return encoded