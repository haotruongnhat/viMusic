import magenta
from magenta.music import sequences_lib
from magenta.music.protobuf import music_pb2
from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
import tensorflow as tf

from .data_lib import PerformanceWithLyrics


class PerformanceWithLyricsExtractor(pipeline.Pipeline):
    """Extracts polyphonic tracks and lyrics from a quantized NoteSequence."""
    
    def __init__(self, min_events, max_events, num_velocity_bins, name=None):
        super(PerformanceWithLyricsExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
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