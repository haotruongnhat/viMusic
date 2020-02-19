#encoder decoder in magenta. It is a sequence, with attribute is one-hot index(or encoding) and label is the value of the next note
#ConditionalEventSequenceEncoderDecoder:
#consists of control event and target event
#For each example i, it is a concatenation of control onehot at i + 1 and target onehot at i, with its label is the value at i + 1
from magenta.models.shared import events_rnn_model
from magenta.music.protobuf.generator_pb2 import GeneratorDetails
from tensorflow.contrib import training as contrib_training
import magenta
#Min note value and max note value to be considered encoded
#Why not 0 and 128 for improv_rnn

class ViMusicConfig(events_rnn_model.EventSequenceRnnConfig):
    #separate the encoder_decoder of performance and encoder_decoder of improv_rnn
    def __init__(self, details,encoder_decoder,  \
    hparams,min_events, max_events,\
    num_velocity_bins=0,steps_per_second=4):

        hparams_dict = {
            'batch_size': 64,
            'dropout_keep_prob': 1.0,
            'learning_rate': 0.001
        }
        hparams_dict.update(hparams.values())

        super(ViMusicConfig,self).__init__(
            details,
            encoder_decoder,
            contrib_training.HParams(**hparams_dict)
        )
            
        self.min_events = min_events
        self.max_events = max_events
        self.num_velocity_bins = num_velocity_bins
        self.steps_per_second = steps_per_second

#parameters for configuration
model_detail = GeneratorDetails(
id='lyrics-based music transformer - gan generator',
description='Mixing model between music transformer and lstm-gan and train with lyrics-based')
data_encoder_decoder = None #have not implemented yet
hparams = contrib_training.HParams(
    batch_size = 64,
    dropout_keep_prob = 1.0,
    learning_rate = 0.001)
min_events = 0
max_events = 64
num_velocity_bins = 32
steps_per_second = 100
###################################33

default_config = ViMusicConfig(model_detail,data_encoder_decoder,hparams,
min_events,max_events,num_velocity_bins,steps_per_second)