import copy

import magenta
from magenta.models.shared import events_rnn_model
import magenta.music as mm
from magenta.common import state_util
from tensorflow.contrib import training as contrib_training
from vimusic_rnn.lib import ViMusicEvent
import functools, collections
from magenta.common import beam_search
import numpy as np
import tensorflow as tf


DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = None

ViMusicControlState = collections.namedtuple(
    'ViMusicControlState', ['current_vi_index', 'current_vi_step'])

class ViMusicRnnModel(events_rnn_model.EventSequenceRnnModel):
    """ViMusicRNNModel"""
    
    def generate_vimusic(self, num_steps, primer_sequence, 
    temperature=1.0, beam_size=1, branch_factor=1, steps_per_iteration=1, 
    control_signal_fns=None, disable_conditioning_fn=None):
        #exist control signal functions
        #from pdb import set_trace ; set_trace()
        if control_signal_fns:
            control_event = tuple(f(0) for f in control_signal_fns)
            if disable_conditioning_fn is not None:
                control_event = (disable_conditioning_fn(0), control_event)
            control_events = [control_event] + [control_event]
            control_state = ViMusicControlState(
                current_vi_index=0, current_vi_step=0
            )
            extend_control_events_callback = functools.partial(
            _extend_control_events, control_signal_fns, disable_conditioning_fn)
        else:
            control_events = None
            control_state = None
            extend_control_events_callback = None

        return self._generate_events(
            num_steps,
            primer_sequence,
            temperature,
            beam_size,
            branch_factor,
            steps_per_iteration,
            control_events,
            control_state,
            extend_control_events_callback
        )

    def vimusic_log_likelihood(self, sequence, control_values, disable_conditioning):
        if control_values:
            control_event = tuple(control_values)
            if disable_conditioning is not None:
                control_event = (disable_conditioning, control_event)
            control_events = [control_event] * len(sequence)
        else:
            control_events = None
        return self._evaluate_log_likelihood([sequence],
                                            control_events=control_events)[0]

def _extend_control_events(control_signal_fns, disable_conditioning_fn,
                           control_target_events, vimusic, control_state):
    idx = control_state.current_vi_index
    step = control_state.current_vi_step
    #from pdb import set_trace ; set_trace()
    for idx, event in enumerate(vimusic):
        if vimusic[idx].event_type == ViMusicEvent.TIME_SHIFT:
            step += vimusic[idx].event_value
        idx += 1

        control_event = tuple(f(step) for f in control_signal_fns)
        if disable_conditioning_fn is not None:
            control_event = (disable_conditioning_fn(step), control_event)
        control_target_events.append(control_event)

    return ViMusicControlState(
        current_vi_index=idx, current_vi_step=step)