import math

from magenta.music import encoder_decoder
from magenta.music.encoder_decoder import EventSequenceEncoderDecoder

from music_transformer.constants import *
import numpy as np

from .data_lib import ViMusicEvent

class ViMusicOneHotSequenceEncoderDecoder(EventSequenceEncoderDecoder):

    def __init__(self,num_velocity_bins):
        self._num_velocity_bins = num_velocity_bins

    @property
    def input_size(self): #number of dimension in a vector
        return (3 + #3 for 8 types of event
        8 # 8 bit for 127 value (of pitch or step shift or velocity change
        )


    @property
    def num_classes(self):
        return int(self.input_size * '1',2) + 1

    def _encode_event(self,event):
        """
        Encode the event into a vector with [input_size] length
        """
        event_type = '{:03b}'.format(event.event_type)
        event_value = '{:08b}'.format(event.event_value)

        code_vector = event_type + event_value

        value = int(code_vector,2)
        
        return value
        
    def _decode_event(self,class_index):
        """
        Decode the vector into an event
        """
        code_vector = '{:0{}b}' \
        .format(class_index,self.input_size)
        event_type_bin = code_vector[:3] #3 bit
        event_value_bin = code_vector[3:] #8 bit

        event_type = int(event_type_bin,2)
        event_value = int(event_value_bin,2)

        return ViMusicEvent(
            event_type,
            event_value
        )

    def _event_to_num_steps(self,event):
        if event.event_type == ViMusicEvent.STEP_SHIFT:
            return event.event_value
        else:
            return 0  


    @property
    def default_event_label(self):
        return self._encode_event(
        ViMusicEvent(
        event_type=ViMusicEvent.STEP_SHIFT,
        event_value=0))

    def events_to_input(self,events,position):
        """
        Input will be a list
        """
        value_list = list('{:0{}b}' \
        .format(self._encode_event(events[position])
        ,self.input_size))
        return [float(x) for x in value_list]

    def events_to_label(self,events,position):
        """
        label will be an integer
        """
        return self._encode_event(events[position])

    def class_index_to_event(self,class_index):
        return self._decode_event(class_index)

    def labels_to_num_steps(self,labels):
        events = []
        for label in labels:
            events.append(self.class_index_to_event(label))
        return sum(self._event_to_num_steps(event) for event in events)
