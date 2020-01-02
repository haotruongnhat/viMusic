#encoder decoder in magenta. It is a sequence, with attribute is one-hot index(or encoding) and label is the value of the next note
#ConditionalEventSequenceEncoderDecoder:
#consists of control event and target event
#For each example i, it is a concatenation of control onehot at i + 1 and target onehot at i, with its label is the value at i + 1
from magenta import music as mm
from magenta.models.shared import events_rnn_model
from magenta.music.protobuf.generator_pb2 import GeneratorDetails
from magenta.music import PitchChordsEncoderDecoder
from magenta.music import OneHotEventSequenceEncoderDecoder
from magenta.music import MelodyOneHotEncoding
from magenta.music import ConditionalEventSequenceEncoderDecoder
from magenta.music import MultipleEventSequenceEncoder
from magenta.music import OptionalEventSequenceEncoder
from tensorflow.contrib import training as contrib_training
import magenta

#Min note value and max note value to be considered encoded
#Why not 0 and 128 for improv_rnn
DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = None

class ViMusicRnnConfig(events_rnn_model.EventSequenceRnnConfig):
    """Stores a configuration for a Performance RNN.

    Attributes:
        num_velocity_bins: Number of velocity bins to use. If 0, don't use velocity
            at all.
        control_signals: List of PerformanceControlSignal objects to use for
            conditioning, or None if not conditioning on anything.
        optional_conditioning: If True, conditioning can be disabled by setting a
            flag as part of the conditioning input.
    """

    #separate the encoder_decoder of performance and encoder_decoder of improv_rnn
    def __init__(self, details, perf_encoder_decoder, improv_encoder_decoder, \
    hparams, num_velocity_bins=0, control_signals=None, \
    optional_conditioning=False,note_performance=False, \
    min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE,\
    transpose_to_key=DEFAULT_TRANSPOSE_TO_KEY):

        if control_signals is not None:
            control_encoder = magenta.music.MultipleEventSequenceEncoder(
            [control.encoder for control in control_signals])
            if optional_conditioning:
                control_encoder = magenta.music.OptionalEventSequenceEncoder(
                control_encoder)
            perf_encoder_decoder = magenta.music.ConditionalEventSequenceEncoderDecoder(
            control_encoder, perf_encoder_decoder)

        encoder_decoder = magenta.music.MultipleEventSequenceEncoder(
            [perf_encoder_decoder, improv_encoder_decoder])

        super(ViMusicRnnConfig, self).__init__(
            details, encoder_decoder, hparams)
        self.num_velocity_bins = num_velocity_bins
        self.control_signals = control_signals
        self.optional_conditioning = optional_conditioning
        self.note_performance = note_performance
        self.min_note = min_note
        self.max_note = max_note
        self.transpose_to_key = transpose_to_key


model_detail = GeneratorDetails(
    id='mix model of vimusic',
    description='Mixing model between performance rnn and improv rnn')

#encoder encoder for the performance rnn
#a. control signal
control_signals = [
    magenta.music.NoteDensityPerformanceControlSignal(
        window_size_seconds=3.0,
        density_bin_ranges=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
    magenta.music.PitchHistogramPerformanceControlSignal(
        window_size_seconds=5.0)
]
#b. encoder_decoder
perf_encoder_decoder = magenta.music.OneHotEventSequenceEncoderDecoder(
    magenta.music.PerformanceOneHotEncoding(num_velocity_bins=32))

#encoder decoder for improv rnn
improv_encoder_decoder = magenta.music.ConditionalEventSequenceEncoderDecoder(
    magenta.music.PitchChordsEncoderDecoder(),
    magenta.music.OneHotEventSequenceEncoderDecoder(
        magenta.music.MelodyOneHotEncoding(
            min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE)))
#hparams
hparams = contrib_training.HParams(
    batch_size=64,
    rnn_layer_sizes=[512, 512, 512],
    dropout_keep_prob=1.0,
    clip_norm=3,
    learning_rate=0.001)

default_vimusic_configuration = ViMusicRnnConfig(
    model_detail,
    perf_encoder_decoder,
    improv_encoder_decoder,
    hparams,
    num_velocity_bins=32,
    control_signals=control_signals,
    optional_conditioning=True,
    note_performance=True
)