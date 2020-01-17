from magenta.music import LeadSheet
from magenta.music import sequences_lib
from magenta.music.protobuf import music_pb2
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
from magenta.music import chord_symbols_lib
from magenta.music.sequences_lib import is_quantized_sequence
import tensorflow as tf
from vimusic_rnn.lib.vimusic_type import ViMusic
import magenta
from magenta.music import infer_dense_chords_for_sequence
import copy

from magenta.music import constants
from magenta.pipelines.note_sequence_pipelines import NoteSequencePipeline
#Leadsheet -> (use ViMusicPipeline) -> MyType -> (use EventSequencePipeline) -> tf.SequenceExample
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
NO_CHORD = constants.NO_CHORD

#defining bass, tenor, alto and soprano range
C2 = 36
C3 = 48
C4 = 60
C5 = 72
C6 = 84
BASS = (C2,C3)
TENOR = (C3,C4)
ALTO = (C4,C5)
SOPRANO = (C5,C6)

class ViMusicEncoderDecoderPipeline(pipeline.Pipeline):
    """A Pipeline that infuse melody, chord and dynamics"""
    def __init__(self, config, name):
        super(ViMusicEncoderDecoderPipeline, self).__init__(
            input_type=ViMusic,
            output_type=tf.train.SequenceExample,
            name=name)
        self._encoder_decoder = config.encoder_decoder
        self._control_signals = config.control_signals
        self._optional_conditioning = config.optional_conditioning

    def transform(self, vimusic):
        # Encode conditional on control signals.
        control_sequences = []
        for control in self._control_signals:
            control_sequences.append(control.extract(vimusic))
        control_sequence = list(zip(*control_sequences))
        # Create two copies, one with and one without conditioning.
        # pylint: disable=g-complex-comprehension
        #remove last event to ensure conditional works properly
        encoded = [
        self._encoder_decoder.encode(
            list(zip([disable] * len(control_sequence), control_sequence)),
            vimusic)
        for disable in [False, True]]
        return encoded

class ViMusicExtractor(pipeline.Pipeline):
    def __init__(self,config,name=None):
        super(ViMusicExtractor,self).__init__(
            input_type=music_pb2.NoteSequence,
            output_type=ViMusic,
            name=name)
        self._min_events = config.min_events
        self._max_events = config.max_events
        self._num_velocity_bins = config.num_velocity_bins

    @staticmethod
    def extract_vimusic(quantized_sequence,
    split_instruments=True,
    num_velocity_bins=0,
    start_step = 0,
    max_steps_truncate = None,
    min_events=None,
    max_events=None):
        sequences_lib.is_absolute_quantized_sequence(quantized_sequence)

        # pylint: disable=g-complex-comprehension
        stats = dict((stat_name, statistics.Counter(stat_name)) for stat_name in
               ['vimusic_discarded_too_short',
                'vimusic_truncated',
                'vimusic_truncated_timewise',
                'vimusic_discarded_more_than_1_program',
                'vimusic_discarded_too_many_time_shift_steps',
                'vimusic_discarded_too_many_duration_steps'])

        steps_per_second = quantized_sequence.quantization_info.steps_per_second

        stats['vimusic_lengths_in_seconds'] = statistics.Histogram(
        'vimusic_lengths_in_seconds',
        [5, 10, 20, 30, 40, 60, 120])

        if split_instruments:
            instruments = set(note.instrument for note in quantized_sequence.notes)
        else:
            instruments = set([None])
            # Allow only 1 program.
            programs = set()
            for note in quantized_sequence.notes:
                programs.add(note.program)
            if len(programs) > 1:
                stats['vimusic_discarded_more_than_1_program'].increment()
                return [], stats.values()

        vimusics = []
        #from pdb import set_trace ; set_trace()
        for instrument in instruments:
            vimusic = ViMusic(quantized_sequence=quantized_sequence,
            start_step=start_step,
            num_velocity_bins=num_velocity_bins,
            instrument=instrument)
            if (max_steps_truncate is not None and
            vimusic.num_steps > max_steps_truncate):
                vimusic.set_length(max_steps_truncate)
                stats['vimusic_truncated_timewise'].increment()

            if (max_events is not None and
            len(vimusic) > max_events):
                vimusic.truncate(max_events)
                stats['vimusic_truncated'].increment()

            if min_events is not None and len(vimusic) < min_events:
                stats['vimusic_discarded_too_short'].increment()
            else:
                vimusics.append(vimusic)  
                stats['vimusic_lengths_in_seconds'].increment(
                vimusic.num_steps // steps_per_second)

        return vimusics, stats.values()

    def transform(self,quantized_sequence):
        split_instruments = False
        start_step = 0
        max_steps_truncate = None

        vimusics, stats_values = ViMusicExtractor.extract_vimusic(quantized_sequence,
        split_instruments=split_instruments,
        num_velocity_bins=self._num_velocity_bins,
        start_step=start_step,
        max_steps_truncate=max_steps_truncate,
        min_events=self._min_events,
        max_events=self._max_events)

        self._set_stats(stats_values)
        return vimusics

class InferChordsPipeline(NoteSequencePipeline):

    def __init__(self,instrument=None,name=None):
        super(InferChordsPipeline,self).__init__(name)
        self.instrument = instrument
    #inder chords from note sequence
    def transform(self, whole_sequence):

        #if there exists chords -> learn using default chord
        if not any([x.text == CHORD_SYMBOL for x in whole_sequence.text_annotations]):
            return [copy.deepcopy(whole_sequence)]

        sequence = copy.deepcopy(whole_sequence)

        #get list of notes
        notes = [ note for note in sequence.notes if not note.is_drum and
        (self.instrument is None or note.instrument == self.instrument)]
        sorted_notes = sorted(notes, key=lambda note: note.start_time)

        # If the sequence is quantized, use quantized steps instead of time.
        if is_quantized_sequence(sequence):
            note_start = lambda note: note.quantized_start_step
            note_end = lambda note: note.quantized_end_step
        else:
            note_start = lambda note: note.start_time
            note_end = lambda note: note.end_time

        # Sort all note start and end events.
        onsets = [
            (note_start(note), idx, False) for idx, note in enumerate(sorted_notes)
        ]
        offsets = [
            (note_end(note), idx, True) for idx, note in enumerate(sorted_notes)
        ]
        events = sorted(onsets + offsets)

        current_time = 0
        current_figure = constants.NO_CHORD
        active_notes = set()

        for time, idx, is_offset in events:
            if time > current_time:
                active_pitches = set(sorted_notes[idx].pitch for idx in active_notes)
                #removing notes that ae in soprano, since it is not in a chord
                #from pdb import set_trace ; set_trace()
                active_pitches = list(filter(lambda x : x < SOPRANO[0],active_pitches))

                # Infer a chord symbol for the active pitches.
                figure = constants.NO_CHORD
                try:
                    figure = chord_symbols_lib.pitches_to_chord_symbol(active_pitches)
                except:
                    print("Skipping chords....")
                    figure = constants.NO_CHORD

                if figure != current_figure:
                    # Add a text annotation to the sequence.
                    text_annotation = sequence.text_annotations.add()
                    text_annotation.text = figure
                    text_annotation.annotation_type = CHORD_SYMBOL
                    if is_quantized_sequence(sequence):
                        text_annotation.time = (
                            current_time * sequence.quantization_info.steps_per_quarter)
                        text_annotation.quantized_step = current_time
                    else:
                        text_annotation.time = current_time

                current_figure = figure

            current_time = time
            if is_offset:
                active_notes.remove(idx)
            else:
                active_notes.add(idx)

        return [sequence]
