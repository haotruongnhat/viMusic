from magenta.music import encoder_decoder
from magenta.music import chord_symbols_lib
from magenta.music import constants

from .vimusic_type import DEFAULT_MAX_SHIFT_STEPS
from .vimusic_type import MIN_MIDI_PITCH
from .vimusic_type import MAX_MIDI_PITCH
from .vimusic_type import ViMusicEvent


class ViMusicOneHotEncoding(encoder_decoder.OneHotEncoding):
    """One-hot encoding for vimusic events."""
    """This one does not include chord, since chord is the control signal"""
    def __init__(self, num_velocity_bins=0,
    max_shift_steps=DEFAULT_MAX_SHIFT_STEPS,
    min_pitch=MIN_MIDI_PITCH,
    max_pitch=MAX_MIDI_PITCH):
        self._event_ranges = [
        (ViMusicEvent.NOTE_ON, min_pitch, max_pitch),
        (ViMusicEvent.NOTE_OFF, min_pitch, max_pitch),
        (ViMusicEvent.TIME_SHIFT, 0, max_shift_steps)
        ]
        #Special case for CHORD_SHIFT
        if num_velocity_bins > 0:
            self._event_ranges.append(
            (ViMusicEvent.VELOCITY, 1, num_velocity_bins))
        self._max_shift_steps = max_shift_steps

    @property
    def num_classes(self):
        return sum(max_value - min_value + 1
               for event_type, min_value, max_value in self._event_ranges)

    @property
    def default_event(self):
        return ViMusicEvent(
        event_type=ViMusicEvent.TIME_SHIFT,
        event_value=self._max_shift_steps)

    def encode_event(self, event):
        offset = 0
        for event_type, min_value, max_value in self._event_ranges:
            if event.event_type == event_type:
                return offset + event.event_value - min_value
            offset += max_value - min_value + 1
        raise ValueError('Unknown event type: %s' % event.event_type)

    def decode_event(self, index):
        offset = 0
        for event_type, min_value, max_value in self._event_ranges:
            if offset <= index <= offset + max_value - min_value:
                return ViMusicEvent(
                event_type=event_type, event_value=min_value + index - offset)
            offset += max_value - min_value + 1

        raise ValueError('Unknown event index: %s' % index)

    def event_to_num_steps(self, event):
        if event.event_type == ViMusicEvent.TIME_SHIFT:
            return event.event_value
        else:
            return 0