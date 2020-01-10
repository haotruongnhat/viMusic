from magenta.music import constants
import math

MAX_MIDI_VELOCITY = constants.MAX_MIDI_VELOCITY
MIN_MIDI_VELOCITY = constants.MIN_MIDI_VELOCITY

def _velocity_bin_size(num_velocity_bins):
	return int(math.ceil(
    (MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1) / num_velocity_bins))


def velocity_to_bin(velocity, num_velocity_bins):
    return ((velocity - MIN_MIDI_VELOCITY) //
	_velocity_bin_size(num_velocity_bins) + 1)


def velocity_bin_to_velocity(velocity_bin, num_velocity_bins):
    return (
    MIN_MIDI_VELOCITY + (velocity_bin - 1) *
    _velocity_bin_size(num_velocity_bins))

