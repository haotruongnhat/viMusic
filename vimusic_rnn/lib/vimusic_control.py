
from __future__ import division

import abc
import copy
import numbers
import magenta

from magenta.music import constants
from magenta.music import encoder_decoder

from vimusic_rnn.lib import ViMusicEvent

NO_CHORD = constants.NO_CHORD
NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
DEFAULT_NOTE_DENSITY = 15.0
DEFAULT_PITCH_HISTOGRAM = [1.0] * NOTES_PER_OCTAVE

class NoteDensityOneHotEncoding(encoder_decoder.OneHotEncoding):
    """One-hot encoding for vimusic note density events.

    Encodes by quantizing note density events. When decoding, always decodes to
    the minimum value for each bin. The first bin starts at zero note density.
    """

    def __init__(self, density_bin_ranges):
        """Initialize a NoteDensityOneHotEncoding.

        Args:
            density_bin_ranges: List of note density (notes per second) bin
                boundaries to use when quantizing. The number of bins will be one
                larger than the list length.
        """
        self._density_bin_ranges = density_bin_ranges

    @property
    def num_classes(self):
        return len(self._density_bin_ranges) + 1

    @property
    def default_event(self):
        return 0.0

    def encode_event(self, event):
        for idx, density in enumerate(self._density_bin_ranges):
            if event < density:
                return idx
        return len(self._density_bin_ranges)

    def decode_event(self, index):
        if index == 0:
            return 0.0
        else:
            return self._density_bin_ranges[index - 1]

class PitchHistogramEncoder(encoder_decoder.EventSequenceEncoderDecoder):
    """An encoder for pitch class histogram sequences."""

    @property
    def input_size(self):
        return NOTES_PER_OCTAVE

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def default_event_label(self):
        raise NotImplementedError

    def events_to_input(self, events, position):
        # Normalize by the total weight.
        total = sum(events[position])
        if total > 0:
            return [count / total for count in events[position]]
        else:
            return [1.0 / NOTES_PER_OCTAVE] * NOTES_PER_OCTAVE

    def events_to_label(self, events, position):
        raise NotImplementedError

    def class_index_to_event(self, class_index, events):
        raise NotImplementedError

class ViMusicControlSignal(object):
    """Control signal used for conditional generation of vimusics.

    The two main components of the control signal (that must be implemented in
    subclasses) are the `extract` method that extracts the control signal values
    from a vimusic object, and the `encoder` class that transforms these
    control signal values into model inputs.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def name(self):
        """Name of the control signal."""
        pass

    @abc.abstractproperty
    def description(self):
        """Description of the control signal."""
        pass

    @abc.abstractmethod
    def validate(self, value):
        """Validate a control signal value."""
        pass

    @abc.abstractproperty
    def default_value(self):
        """Default value of the (unencoded) control signal."""
        pass

    @abc.abstractproperty
    def encoder(self):
        """Instantiated encoder object for the control signal."""
        pass

    @abc.abstractmethod
    def extract(self, vimusic):
        """Extract a sequence of control values from a vimusic object.

        Args:
        vimusic: The vimusic object from which to extract control signal
        values.

        Returns:
        A sequence of control signal values the same length as `vimusic`.
        """
        pass

class NoteDensityViMusicControlSignal(ViMusicControlSignal):
    """density (notes / chord per second) vimusic control signal."""

    name = 'note_density_control_signal'
    description = 'Desired number of notes per second.'

    def __init__(self, window_size_seconds, density_bin_ranges):
        """Initialize a NoteDensityvimusicControlSignal.

        Args:
        window_size_seconds: The size of the window, in seconds, used to compute
            note density (notes per second).
        density_bin_ranges: List of note density (notes per second) bin boundaries
            to use when quantizing. The number of bins will be one larger than the
            list length.
        """
        self._window_size_seconds = window_size_seconds
        self._density_bin_ranges = density_bin_ranges
        self._encoder = encoder_decoder.OneHotEventSequenceEncoderDecoder(
            NoteDensityOneHotEncoding(density_bin_ranges))

    def validate(self, value):
        return isinstance(value, numbers.Number) and value >= 0.0

    @property
    def default_value(self):
        return DEFAULT_NOTE_DENSITY

    @property
    def encoder(self):
        return self._encoder

    def extract(self, vimusic):
        """Computes note density at every event in a vimusic.
        
        Args:
        vimusic: A vimusic object for which to compute a note density
            sequence.

        Returns:
        A list of note densities of the same length as `vimusic`, with each
        entry equal to the note density in the window starting at the
        corresponding vimusic event time.
        """
        window_size_steps = int(round(
        self._window_size_seconds * vimusic.steps_per_second))

        prev_event_type = None
        prev_density = 0.0

        density_sequence = []
        for i, event in enumerate(vimusic.get_notes()):
            if (prev_event_type is not None and
            prev_event_type != ViMusicEvent.TIME_SHIFT):
                # The previous event didn't move us forward in time, so the note density
                # here should be the same.
                density_sequence.append(prev_density)
                prev_event_type = event.event_type
                continue

            j = i
            step_offset = 0
            note_count = 0

            # Count the number of note-on events within the window.
            while step_offset < window_size_steps and j < len(vimusic.get_notes()):
                if vimusic.get_notes()[j].event_type == ViMusicEvent.NOTE_ON:
                    note_count += 1
                elif vimusic.get_notes()[j].event_type == ViMusicEvent.TIME_SHIFT:
                    step_offset += vimusic.get_notes()[j].event_value
                j += 1

            # If we're near the end of the vimusic, part of the window will
            # necessarily be empty; we don't include this part of the window when
            # calculating note density.
            actual_window_size_steps = min(step_offset, window_size_steps)
            if actual_window_size_steps > 0:
                density = (
                note_count * vimusic.steps_per_second /
                actual_window_size_steps)
            else:
                density = 0.0

            density_sequence.append(density)

            prev_event_type = event.event_type
            prev_density = density

        #padding last density control signal
        #density_sequence.append(self.default_value)

        return density_sequence

class PitchChordViMusicControlSignal(ViMusicControlSignal):
    """density (notes / chord per second) vimusic control signal."""

    name = 'chord_control_signal'
    description = 'Define the current chord on each event'

    def __init__(self):
        """Initialize a NoteDensityvimusicControlSignal.

        Args:
        window_size_seconds: The size of the window, in seconds, used to compute
            note density (notes per second).
        density_bin_ranges: List of note density (notes per second) bin boundaries
            to use when quantizing. The number of bins will be one larger than the
            list length.
        """
        self._encoder = magenta.music.PitchChordsEncoderDecoder()

    def validate(self, value):
        return isinstance(value, numbers.Number) and value >= 0.0

    @property
    def default_value(self):
        return NO_CHORD

    @property
    def encoder(self):
        return self._encoder

    def extract(self, vimusic):
        """Compute pitch chord at every events"""
        prev_value = NO_CHORD
        chord_sequence = []

        for i, event in enumerate(vimusic.get_chords()):
            if event.event_type == ViMusicEvent.CHORD_ON:
                prev_value = event.event_value
            elif event.event_type == ViMusicEvent.CHORD_OFF:
                prev_value = NO_CHORD
            chord_sequence.append(prev_value)

        #chord_sequence.append(self.default_value)

        return chord_sequence

class PitchHistogramViMusicControlSignal(ViMusicControlSignal):
    """Pitch class histogram performance control signal."""

    name = 'pitch_class_histogram'
    description = 'Desired weight for each for each of the 12 pitch classes.'

    def __init__(self, window_size_seconds, prior_count=0.01):
        """Initializes a PitchHistogramPerformanceControlSignal.

        Args:
        window_size_seconds: The size of the window, in seconds, used to compute
            each histogram.
        prior_count: A prior count to smooth the resulting histograms. This value
            will be added to the actual pitch class counts.
        """
        self._window_size_seconds = window_size_seconds
        self._prior_count = prior_count
        self._encoder = PitchHistogramEncoder()

    @property
    def default_value(self):
        return DEFAULT_PITCH_HISTOGRAM

    def validate(self, value):
        return (isinstance(value, list) and len(value) == NOTES_PER_OCTAVE and
                all(isinstance(a, numbers.Number) for a in value))

    @property
    def encoder(self):
        return self._encoder

    def extract(self, vimusic):
        """Computes local pitch class histogram at every event in a performance.

        Args:
        performance: A Performance object for which to compute a pitch class
            histogram sequence.

        Returns:
        A list of pitch class histograms the same length as `performance`, where
        each pitch class histogram is a length-12 list of float values summing to
        one.
        """
        window_size_steps = int(round(
            self._window_size_seconds * vimusic.steps_per_second))

        prev_event_type = None
        prev_histogram = self.default_value

        base_active_pitches = set()
        histogram_sequence = []

        for i, event in enumerate(vimusic.get_notes()):
            # Maintain the base set of active pitches.
            if event.event_type == ViMusicEvent.NOTE_ON:
                base_active_pitches.add(event.event_value)
            elif event.event_type == ViMusicEvent.NOTE_OFF:
                base_active_pitches.discard(event.event_value)

            if (prev_event_type is not None and
                prev_event_type != ViMusicEvent.TIME_SHIFT):
                # The previous event didn't move us forward in time, so the histogram
                # here should be the same.
                histogram_sequence.append(prev_histogram)
                prev_event_type = event.event_type
                continue

            j = i
            step_offset = 0

            active_pitches = copy.deepcopy(base_active_pitches)
            histogram = [self._prior_count] * NOTES_PER_OCTAVE

            # Count the total duration of each pitch class within the window.
            while step_offset < window_size_steps and j < len(vimusic.get_notes()):
                if vimusic.get_notes()[j].event_type == ViMusicEvent.NOTE_ON:
                    active_pitches.add(vimusic.get_notes()[j].event_value)
                elif vimusic.get_notes()[j].event_type == ViMusicEvent.NOTE_OFF:
                    active_pitches.discard(vimusic.get_notes()[j].event_value)
                elif vimusic.get_notes()[j].event_type == ViMusicEvent.TIME_SHIFT:
                    for pitch in active_pitches:
                        histogram[pitch % NOTES_PER_OCTAVE] += (
                        vimusic.get_notes()[j].event_value / vimusic.steps_per_second)
                    step_offset += vimusic.get_notes()[j].event_value
                j += 1

            histogram_sequence.append(histogram)

            prev_event_type = event.event_type
            prev_histogram = histogram
        #histogram_sequence.append(self.default_value)
        return histogram_sequence