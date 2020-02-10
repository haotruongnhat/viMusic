import magenta
from magenta.music import sequences_lib
from music_transformer.music.protobuf import vimusic_pb2
from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
import tensorflow as tf
import copy

from music_transformer.data_pipeline import PerformanceWithLyrics


class PerformanceWithLyricsExtractor(pipeline.Pipeline):
    """Extracts polyphonic tracks and lyrics from a quantized NoteSequence."""
    
    def __init__(self, min_events, max_events, num_velocity_bins, name=None):
        super(PerformanceWithLyricsExtractor, self).__init__(
        input_type=vimusic_pb2.NoteSequence,
        output_type=PerformanceWithLyrics,
        name=name)
        self._min_events = min_events
        self._max_events = max_events
        self._num_velocity_bins = num_velocity_bins

    def transform(self, quantized_sequence):
        """It's extraction time :>"""
        """
        Condition:
            sequence must be quantized
            sequence must be only in one time signature
        """
        #Validate quantized_sequence
        sequences_lib.assert_is_quantized_sequence(quantized_sequence)

        # pylint: disable=g-complex-comprehension
        #Getting stats
        stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
        ['performances_truncated',
         'performances_discarded'])

        #always allow more than 1 program
        instruments = set(note.instrument for note in quantized_sequence.notes)

        performances = []

        for instrument in instruments:
            # Translate the quantized sequence into a Performance.
            performance = PerformanceWithLyrics(quantized_sequence, start_step=0,
                                num_velocity_bins=self._num_velocity_bins,
                                instrument=instrument)
            if (self._max_events is not None and
            len(performance) > self._max_events):
                performance.truncate(self._max_events)
                stats['performances_truncated'].increment()

            if self._min_events is not None and len(performance) < self._min_events:
                stats['performances_discarded'].increment()
                
            performances.append(performance)

        return performances, stats.values()
        

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

class ViMusicEncoderPipeline(pipeline.Pipeline):
    def __init__(self,name=None):
        super(ViMusicEncoderPipeline, self).__init__(
        input_type=PerformanceWithLyrics,
        output_type=tf.train.SequenceExample,
        name=name)

    def transform(self,performance_with_lyrics):
        return [None]