#encoder decoder in magenta. It is a sequence, with attribute is one-hot index(or encoding) and label is the value of the next note
#ConditionalEventSequenceEncoderDecoder:
#consists of control event and target event
#For each example i, it is a concatenation of control onehot at i + 1 and target onehot at i, with its label is the value at i + 1
from magenta.models.shared import events_rnn_model
from magenta.music.protobuf.generator_pb2 import GeneratorDetails
from tensorflow.contrib import training as contrib_training
import magenta
from vimusic_rnn.lib import NoteDensityViMusicControlSignal
from vimusic_rnn.lib import PitchChordViMusicControlSignal
from vimusic_rnn.lib import PitchHistogramViMusicControlSignal
from vimusic_rnn.lib import ViMusicOneHotEncoding
#Min note value and max note value to be considered encoded
#Why not 0 and 128 for improv_rnn

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
    def __init__(self, details, encoder_decoder,  \
    hparams,min_events, max_events, \
    num_velocity_bins=0, control_signals=None, \
    optional_conditioning=False):

        hparams_dict = {
            'batch_size': 64,
            'rnn_layer_sizes': [128, 128],
            'dropout_keep_prob': 1.0,
            'attn_length': 0,
            'clip_norm': 3,
            'learning_rate': 0.001,
            'residual_connections': False,
            'use_cudnn': False
        }
        hparams_dict.update(hparams.values())

        if control_signals is not None:
            control_encoder = magenta.music.MultipleEventSequenceEncoder(
            [control.encoder for control in control_signals])
            if optional_conditioning:
                control_encoder = magenta.music.OptionalEventSequenceEncoder(
                control_encoder)
            encoder_decoder = magenta.music.ConditionalEventSequenceEncoderDecoder(
            control_encoder, encoder_decoder)


        super(ViMusicRnnConfig,self).__init__(
            details,
            encoder_decoder,
            contrib_training.HParams(**hparams_dict)
        )
            
        self.min_events = min_events
        self.max_events = max_events
        self.num_velocity_bins = num_velocity_bins
        self.control_signals = control_signals
        self.optional_conditioning = optional_conditioning

#model_detail
model_detail = GeneratorDetails(
    id='mix model of vimusic',
    description='Mixing model between performance rnn and improv rnn')

#encoder_decoder
encoder_decoder = magenta.music.OneHotEventSequenceEncoderDecoder(
    ViMusicOneHotEncoding(num_velocity_bins=32))

#hparams
hparams = contrib_training.HParams(
    batch_size=32,
    rnn_layer_sizes=[256, 256, 256],
    dropout_keep_prob=0.8,
    clip_norm=3,
    learning_rate=0.005)

min_events = 0
max_events = 512

#num_velocity_bins
num_velocity_bins = 32

#control_signals
control_signals = [
    NoteDensityViMusicControlSignal(
        window_size_seconds=5.0,
        density_bin_ranges=[1.0, 2.0, 4.0, 8.0, 16.0]),
    PitchHistogramViMusicControlSignal(
        window_size_seconds=3.0),
    PitchChordViMusicControlSignal()]

#optional_conditioning
optional_conditioning = True

default_vimusic_configuration = ViMusicRnnConfig(
    model_detail,
    encoder_decoder,
    hparams,
    min_events,
    max_events,
    num_velocity_bins,
    control_signals,
    optional_conditioning
)