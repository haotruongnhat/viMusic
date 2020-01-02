from magenta.music import LeadSheet
from magenta.music import sequences_lib
from magenta.music.protobuf import music_pb2
from magenta.pipelines import pipeline
from magenta.pipelines import statistics
import tensorflow as tf
from magenta.music.performance_lib import BasePerformance

from magenta.music import constants
#Leadsheet -> (use ViMusicPipeline) -> MyType -> (use EventSequencePipeline) -> tf.SequenceExample
class ViMusicNoteSequencePipeline(pipeline.Pipeline):
    """Superclass for pipelines that input and output NoteSequences."""

    def __init__(self, name=None):
        """Construct a NoteSequencePipeline. Should only be called by subclasses.
        This class is a pipeline for two models: improv and performance

        Args:
            name: Pipeline name.
        """
        super(ViMusicNoteSequencePipeline, self).__init__(
            input_type={"improv" : music_pb2.NoteSequence, \
                "perf" : music_pb2.NoteSequence},
            output_type={"improv" : music_pb2.NoteSequence, \
                "perf" : music_pb2.NoteSequence},
            name=name)

class ViMusicEncoderDecoderPipeline(pipeline.Pipeline):
    """A Pipeline that infuse melody, chord and dynamics"""
    def __init__(self, config, name):
        super(ViMusicEncoderDecoderPipeline, self).__init__(
            input_type={"improv" : LeadSheet,
            "perf" : BasePerformance},
            output_type=tf.train.SequenceExample,
            name=name)
        self._encoder_decoder = config.encoder_decoder
        self._control_signals = config.control_signals
        self._optional_conditioning = config.optional_conditioning

    #this is what we focus on, infusing melody, chord and dynamics
    def transform(self, data_dict):
        from pdb import set_trace ; set_trace()
        performance = data_dict["perf"]
        if self._control_signals:
            # Encode conditional on control signals.
            control_sequences = []
            for control in self._control_signals:
                control_sequences.append(control.extract(data_dict["perf"]))
            control_sequence = zip(*control_sequences)
            if self._optional_conditioning:
                # Create two copies, one with and one without conditioning.
                # pylint: disable=g-complex-comprehension
                #from pdb import set_trace ; set_trace()
                control_sequence = list(control_sequence)
                encoded = [
                    self._encoder_decoder.encode(
                        list(zip([disable] * len(control_sequence), control_sequence)),
                        performance)
                    for disable in [False, True]]
                # pylint: enable=g-complex-comprehension
            else:
                control_sequence = list(control_sequence)
                encoded = [self._encoder_decoder.encode(
                    control_sequence, performance)]
        else:
            # Encode unconditional.
            encoded = [self._encoder_decoder.encode(performance)]
        return encoded

class ViMusicQuantizer(ViMusicNoteSequencePipeline):
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
        super(ViMusicQuantizer, self).__init__(name=name)
        self._steps_per_quarter = steps_per_quarter
        self._steps_per_second = steps_per_second

    def transform(self, note_sequence_dict):
        #due to performance requiring absolute quantizing
        result = dict()
        result["improv"] = []
        result["perf"] = []
        for key in note_sequence_dict:
            try:
                if key == "improv":
                    quantized_sequence = sequences_lib.quantize_note_sequence(
                    note_sequence_dict[key], self._steps_per_quarter)
                elif key == "perf":
                    quantized_sequence = sequences_lib.quantize_note_sequence_absolute(
                    note_sequence_dict[key], self._steps_per_second)
                result[key] = [quantized_sequence]
            except sequences_lib.MultipleTimeSignatureError as e:
                tf.logging.warning('Multiple time signatures in NoteSequence %s: %s',
                                note_sequence_dict[key].filename, e)
                self._set_stats([statistics.Counter(
                'sequences_discarded_because_multiple_time_signatures', 1)])
                return result
            except sequences_lib.MultipleTempoError as e:
                tf.logging.warning('Multiple tempos found in NoteSequence %s: %s',
                                note_sequence_dict[key].filename, e)
                self._set_stats([statistics.Counter(
                'sequences_discarded_because_multiple_tempos', 1)])
                return result
            except sequences_lib.BadTimeSignatureError as e:
                tf.logging.warning('Bad time signature in NoteSequence %s: %s',
                                note_sequence_dict[key].filename, e)
                self._set_stats([statistics.Counter(
                'sequences_discarded_because_bad_time_signature', 1)])
                return result
        return result

class ViMusicGetter(pipeline.Pipeline):
    """Dumb class for getting value based on key"""

    def __init__(self, key,name=None):
        super(ViMusicGetter, self).__init__(
            input_type={"improv" : music_pb2.NoteSequence, \
                "perf" : music_pb2.NoteSequence},
            output_type=music_pb2.NoteSequence,
            name=name)
        self.key = key
    
    def transform(self, note_sequence_dict):
        try:
            getter = note_sequence_dict[self.key]
        except KeyError:
            tf.logging.warning('Bad key')
            return []
        return [getter]