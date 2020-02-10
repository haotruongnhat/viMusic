from __future__ import division

import abc
import collections
import math
import os

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib
from music_transformer.music import music_pb2
from magenta.music import note_sequence_io
import tensorflow as tf

MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH

MAX_MIDI_VELOCITY = constants.MAX_MIDI_VELOCITY
MIN_MIDI_VELOCITY = constants.MIN_MIDI_VELOCITY
MAX_NUM_VELOCITY_BINS = MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1

STANDARD_PPQ = constants.STANDARD_PPQ

DEFAULT_MAX_SHIFT_STEPS = 100
DEFAULT_MAX_SHIFT_QUARTERS = 4

DEFAULT_PROGRAM = 0


class PerformanceWithLyricsEvent(object):
	"""Class for storing events in a performance."""

	# Start of a new note.
	NOTE_ON = 1
	# End of a note.
	NOTE_OFF = 2
	# Shift time forward.
	TIME_SHIFT = 3
	# Change current velocity.
	VELOCITY = 4
	# Duration of preceding NOTE_ON.
	# For Note-based encoding, used instead of NOTE_OFF events.
	DURATION = 5
	#Start of a new word
	WORD_ON = 6
	#End of a new word
	WORD_OFF = 7

	def __init__(self, event_type, event_value):
		if event_type in (PerformanceWithLyricsEvent.NOTE_ON, PerformanceWithLyricsEvent.NOTE_OFF):
			if not MIN_MIDI_PITCH <= event_value <= MAX_MIDI_PITCH:
				raise ValueError('Invalid pitch value: %s' % event_value)
		elif event_type == PerformanceWithLyricsEvent.TIME_SHIFT:
			if not 0 <= event_value:
				raise ValueError('Invalid time shift value: %s' % event_value)
		elif event_type == PerformanceWithLyricsEvent.DURATION:
			if not 1 <= event_value:
				raise ValueError('Invalid duration value: %s' % event_value)
		elif event_type == PerformanceWithLyricsEvent.VELOCITY:
			if not 1 <= event_value <= MAX_NUM_VELOCITY_BINS:
				raise ValueError('Invalid velocity value: %s' % event_value)
		elif event_type in (PerformanceWithLyricsEvent.WORD_ON, PerformanceWithLyricsEvent.WORD_OFF):
			if not isinstance(event_value,str):
				raise ValueError('Invalid string value : %s' % event_value)
		else:
			raise ValueError('Invalid event type: %s' % event_type)

		self.event_type = event_type
		self.event_value = event_value

	def __repr__(self):
		return 'PerformanceWithLyricsEvent(%r, %r)' % (self.event_type, self.event_value)

	def __eq__(self, other):
		if not isinstance(other, PerformanceWithLyricsEvent):
			return False
		return (self.event_type == other.event_type and self.event_value == other.event_value)

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


def _program_and_is_drum_from_sequence(sequence, instrument=None):
    """Check if sequence contain notes produced by drum"""
    notes = [note for note in sequence.notes
    if instrument is None or note.instrument == instrument]
    # Only set program for non-drum tracks.
    if all(note.is_drum for note in notes):
        is_drum = True
        program = None
    elif all(not note.is_drum for note in notes):
        is_drum = False
        programs = set(note.program for note in notes)
        program = programs.pop() if len(programs) == 1 else None
    else:
        is_drum = None
        program = None
    return program, is_drum

'''
class PerformanceWithLyrics(events_lib.EventSequence):


  	def __init__(self, quantized_sequence,
	start_step, num_velocity_bins, instrument=None):
    	if num_velocity_bins > MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1:
      		raise ValueError(
          	'Number of velocity bins is too large: %d' % num_velocity_bins)

		self._start_step = start_step
		self._num_velocity_bins = num_velocity_bins
		self._max_shift_steps = max_shift_steps
		self._program = program
		self._is_drum = is_drum

  @property
  def start_step(self):
    return self._start_step

  @property
  def max_shift_steps(self):
    return self._max_shift_steps

  @property
  def program(self):
    return self._program

  @property
  def is_drum(self):
    return self._is_drum

  def _append_steps(self, num_steps):
    """Adds steps to the end of the sequence."""
    if (self._events and
        self._events[-1].event_type == PerformanceWithLyricsEvent.TIME_SHIFT and
        self._events[-1].event_value < self._max_shift_steps):
      # Last event is already non-maximal time shift. Increase its duration.
      added_steps = min(num_steps,
                        self._max_shift_steps - self._events[-1].event_value)
      self._events[-1] = PerformanceWithLyricsEvent(
          PerformanceWithLyricsEvent.TIME_SHIFT,
          self._events[-1].event_value + added_steps)
      num_steps -= added_steps

    while num_steps >= self._max_shift_steps:
      self._events.append(
          PerformanceWithLyricsEvent(event_type=PerformanceWithLyricsEvent.TIME_SHIFT,
                           event_value=self._max_shift_steps))
      num_steps -= self._max_shift_steps

    if num_steps > 0:
      self._events.append(
          PerformanceWithLyricsEvent(event_type=PerformanceWithLyricsEvent.TIME_SHIFT,
                           event_value=num_steps))

  def _trim_steps(self, num_steps):
    """Trims a given number of steps from the end of the sequence."""
    steps_trimmed = 0
    while self._events and steps_trimmed < num_steps:
      if self._events[-1].event_type == PerformanceWithLyricsEvent.TIME_SHIFT:
        if steps_trimmed + self._events[-1].event_value > num_steps:
          self._events[-1] = PerformanceWithLyricsEvent(
              event_type=PerformanceWithLyricsEvent.TIME_SHIFT,
              event_value=(self._events[-1].event_value -
                           num_steps + steps_trimmed))
          steps_trimmed = num_steps
        else:
          steps_trimmed += self._events[-1].event_value
          self._events.pop()
      else:
        self._events.pop()

  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads with time shifts to make the
    sequence the specified length. If it is too long, it will be truncated to
    the requested length.

    Args:
      steps: How many quantized steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if from_left:
      raise NotImplementedError('from_left is not supported')

    if self.num_steps < steps:
      self._append_steps(steps - self.num_steps)
    elif self.num_steps > steps:
      self._trim_steps(self.num_steps - steps)

    assert self.num_steps == steps

  def append(self, event):
    """Appends the event to the end of the sequence.

    Args:
      event: The performance event to append to the end.

    Raises:
      ValueError: If `event` is not a valid performance event.
    """
    if not isinstance(event, PerformanceWithLyricsEvent):
      raise ValueError('Invalid performance event: %s' % event)
    self._events.append(event)

  def truncate(self, num_events):
    """Truncates this Performance to the specified number of events.

    Args:
      num_events: The number of events to which this performance will be
          truncated.
    """
    self._events = self._events[:num_events]

  def __len__(self):
    """How many events are in this sequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __iter__(self):
    """Return an iterator over the events in this sequence."""
    return iter(self._events)

  def __str__(self):
    strs = []
    for event in self:
      if event.event_type == PerformanceWithLyricsEvent.NOTE_ON:
        strs.append('(%s, ON)' % event.event_value)
      elif event.event_type == PerformanceWithLyricsEvent.NOTE_OFF:
        strs.append('(%s, OFF)' % event.event_value)
      elif event.event_type == PerformanceWithLyricsEvent.TIME_SHIFT:
        strs.append('(%s, SHIFT)' % event.event_value)
      elif event.event_type == PerformanceWithLyricsEvent.VELOCITY:
        strs.append('(%s, VELOCITY)' % event.event_value)
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)
    return '\n'.join(strs)

  @property
  def end_step(self):
    return self.start_step + self.num_steps

  @property
  def num_steps(self):
    """Returns how many steps long this sequence is.

    Returns:
      Length of the sequence in quantized steps.
    """
    steps = 0
    for event in self:
      if event.event_type == PerformanceWithLyricsEvent.TIME_SHIFT:
        steps += event.event_value
    return steps

  @property
  def steps(self):
    """Return a Python list of the time step at each event in this sequence."""
    step = self.start_step
    result = []
    for event in self:
      result.append(step)
      if event.event_type == PerformanceWithLyricsEvent.TIME_SHIFT:
        step += event.event_value
    return result

  @staticmethod
  def _from_quantized_sequence(quantized_sequence, start_step,
                               num_velocity_bins, max_shift_steps,
                               instrument=None):
    """Extract a list of events from the given quantized NoteSequence object.

    Within a step, new pitches are started with NOTE_ON and existing pitches are
    ended with NOTE_OFF. TIME_SHIFT shifts the current step forward in time.
    VELOCITY changes the current velocity value that will be applied to all
    NOTE_ON events.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start converting the sequence at this time step.
      num_velocity_bins: Number of velocity bins to use. If 0, velocity events
          will not be included at all.
      max_shift_steps: Maximum number of steps for a single time-shift event.
      instrument: If not None, extract only the specified instrument. Otherwise,
          extract all instruments into a single event list.

    Returns:
      A list of events.
    """
    notes = [note for note in quantized_sequence.notes
             if note.quantized_start_step >= start_step
             and (instrument is None or note.instrument == instrument)]
    sorted_notes = sorted(notes, key=lambda note: (note.start_time, note.pitch))

    # Sort all note start and end events.
    onsets = [(note.quantized_start_step, idx, False)
              for idx, note in enumerate(sorted_notes)]
    offsets = [(note.quantized_end_step, idx, True)
               for idx, note in enumerate(sorted_notes)]
    note_events = sorted(onsets + offsets)

    current_step = start_step
    current_velocity_bin = 0
    performance_events = []

    for step, idx, is_offset in note_events:
      if step > current_step:
        # Shift time forward from the current step to this event.
        while step > current_step + max_shift_steps:
          # We need to move further than the maximum shift size.
          performance_events.append(
              PerformanceWithLyricsEvent(event_type=PerformanceWithLyricsEvent.TIME_SHIFT,
                               event_value=max_shift_steps))
          current_step += max_shift_steps
        performance_events.append(
            PerformanceWithLyricsEvent(event_type=PerformanceWithLyricsEvent.TIME_SHIFT,
                             event_value=int(step - current_step)))
        current_step = step

      # If we're using velocity and this note's velocity is different from the
      # current velocity, change the current velocity.
      if num_velocity_bins:
        velocity_bin = velocity_to_bin(
            sorted_notes[idx].velocity, num_velocity_bins)
        if not is_offset and velocity_bin != current_velocity_bin:
          current_velocity_bin = velocity_bin
          performance_events.append(
              PerformanceWithLyricsEvent(event_type=PerformanceWithLyricsEvent.VELOCITY,
                               event_value=current_velocity_bin))

      # Add a performance event for this note on/off.
      event_type = (
          PerformanceWithLyricsEvent.NOTE_OFF if is_offset else PerformanceWithLyricsEvent.NOTE_ON)
      performance_events.append(
          PerformanceWithLyricsEvent(event_type=event_type,
                           event_value=sorted_notes[idx].pitch))

    return performance_events

  @abc.abstractmethod
  def to_sequence(self, velocity, instrument, program, max_note_duration=None):
    """Converts the Performance to NoteSequence proto.

    Args:
      velocity: MIDI velocity to give each note. Between 1 and 127 (inclusive).
          If the performance contains velocity events, those will be used
          instead.
      instrument: MIDI instrument to give each note.
      program: MIDI program to give each note, or None to use the program
          associated with the Performance (or the default program if none
          exists).
      max_note_duration: Maximum note duration in seconds to allow. Notes longer
          than this will be truncated. If None, notes can be any length.

    Returns:
      A NoteSequence proto.
    """
    pass

  def _to_sequence(self, seconds_per_step, velocity, instrument, program,
                   max_note_duration=None):
    sequence_start_time = self.start_step * seconds_per_step

    sequence = music_pb2.NoteSequence()
    sequence.ticks_per_quarter = STANDARD_PPQ

    step = 0

    if program is None:
      # Use program associated with the performance (or default program).
      program = self.program if self.program is not None else DEFAULT_PROGRAM
    is_drum = self.is_drum if self.is_drum is not None else False

    # Map pitch to list because one pitch may be active multiple times.
    pitch_start_steps_and_velocities = collections.defaultdict(list)
    for i, event in enumerate(self):
      if event.event_type == PerformanceWithLyricsEvent.NOTE_ON:
        pitch_start_steps_and_velocities[event.event_value].append(
            (step, velocity))
      elif event.event_type == PerformanceWithLyricsEvent.NOTE_OFF:
        if not pitch_start_steps_and_velocities[event.event_value]:
          tf.logging.debug(
              'Ignoring NOTE_OFF at position %d with no previous NOTE_ON' % i)
        else:
          # Create a note for the pitch that is now ending.
          pitch_start_step, pitch_velocity = pitch_start_steps_and_velocities[
              event.event_value][0]
          pitch_start_steps_and_velocities[event.event_value] = (
              pitch_start_steps_and_velocities[event.event_value][1:])
          if step == pitch_start_step:
            tf.logging.debug(
                'Ignoring note with zero duration at step %d' % step)
            continue
          note = sequence.notes.add()
          note.start_time = (pitch_start_step * seconds_per_step +
                             sequence_start_time)
          note.end_time = step * seconds_per_step + sequence_start_time
          if (max_note_duration and
              note.end_time - note.start_time > max_note_duration):
            note.end_time = note.start_time + max_note_duration
          note.pitch = event.event_value
          note.velocity = pitch_velocity
          note.instrument = instrument
          note.program = program
          note.is_drum = is_drum
          if note.end_time > sequence.total_time:
            sequence.total_time = note.end_time
      elif event.event_type == PerformanceWithLyricsEvent.TIME_SHIFT:
        step += event.event_value
      elif event.event_type == PerformanceWithLyricsEvent.VELOCITY:
        assert self._num_velocity_bins
        velocity = velocity_bin_to_velocity(
            event.event_value, self._num_velocity_bins)
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)

    # There could be remaining pitches that were never ended. End them now
    # and create notes.
    for pitch in pitch_start_steps_and_velocities:
      for pitch_start_step, pitch_velocity in pitch_start_steps_and_velocities[
          pitch]:
        if step == pitch_start_step:
          tf.logging.debug(
              'Ignoring note with zero duration at step %d' % step)
          continue
        note = sequence.notes.add()
        note.start_time = (pitch_start_step * seconds_per_step +
                           sequence_start_time)
        note.end_time = step * seconds_per_step + sequence_start_time
        if (max_note_duration and
            note.end_time - note.start_time > max_note_duration):
          note.end_time = note.start_time + max_note_duration
        note.pitch = pitch
        note.velocity = pitch_velocity
        note.instrument = instrument
        note.program = program
        note.is_drum = is_drum
        if note.end_time > sequence.total_time:
          sequence.total_time = note.end_time

    return sequence
'''

###########Functions to generate dataset from raw
def convert_midi(root_dir, sub_dir, full_file_path):
    """Converts a midi file to a sequence proto.

    Args:
        root_dir: A string specifying the root directory for the files being
            converted.
        sub_dir: The directory being converted currently.
        full_file_path: the full path to the file to convert.

    Returns:
        Either a NoteSequence proto or None if the file could not be converted.
    """
    """
    try:
        sequence = midi_io.midi_to_sequence_proto(
        tf.gfile.GFile(full_file_path, 'rb').read())
    except midi_io.MIDIConversionError as e:
        tf.logging.warning(
        'Could not parse MIDI file %s. It will be skipped. Error was: %s',
        full_file_path, e)
        return None
    sequence.collection_name = os.path.basename(root_dir)
    sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
    sequence.id = note_sequence_io.generate_note_sequence_id(
    sequence.filename, sequence.collection_name, 'midi')
    tf.logging.info('Converted MIDI file %s.', full_file_path)
    return sequence
    """
    #TODO: Find a way to read midi
    return None

def convert_musicxml(root_dir, sub_dir, full_file_path):
    """Converts a musicxml file to a sequence proto.
    Args:
        root_dir: A string specifying the root directory for the files being
            converted.
        sub_dir: The directory being converted currently.
        full_file_path: the full path to the file to convert.
    Returns:
        Either a NoteSequence proto or None if the file could not be converted.
    """
    try:
        sequence = musicxml_reader.musicxml_file_to_sequence_proto(full_file_path)
    except musicxml_reader.MusicXMLConversionError as e:
        tf.logging.warning(
            'Could not parse MusicXML file %s. It will be skipped. Error was: %s',
            full_file_path, e)
        return None
    sequence.collection_name = os.path.basename(root_dir)
    sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
    sequence.id = note_sequence_io.generate_note_sequence_id(
        sequence.filename, sequence.collection_name, 'musicxml')
    tf.logging.info('Converted MusicXML file %s.', full_file_path)
    return sequence

def convert_files(root_dir, sub_dir, writer, recursive=False):
    dir_to_convert = os.path.join(root_dir, sub_dir)
    tf.logging.info("Converting files in '%s'.", dir_to_convert)
    files_in_dir = tf.gfile.ListDirectory(os.path.join(dir_to_convert))
    recurse_sub_dirs = []
    written_count = 0
    for file_in_dir in files_in_dir:
        tf.logging.log_every_n(tf.logging.INFO, '%d files converted.',
        1000, written_count)
        full_file_path = os.path.join(dir_to_convert, file_in_dir)
        if full_file_path.lower().endswith('mxl'):
            sequence = None
            try:
                sequence = convert_musicxml(root_dir, sub_dir, full_file_path)
            except Exception as exc:  # pylint: disable=broad-except
                tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
                continue
            if sequence:
                writer.write(sequence)
        elif full_file_path.lower().endswith('mid') or full_file_path.lower().endswith('midi'):
            sequence = None
            try:
                sequence = convert_midi(root_dir, sub_dir, full_file_path)
            except Exception as exc:  # pylint: disable=broad-except
                tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
                continue
            if sequence:
                writer.write(sequence)
        else:
            if recursive and tf.gfile.IsDirectory(full_file_path):
                recurse_sub_dirs.append(os.path.join(sub_dir, file_in_dir))
            else:
                tf.logging.warning(
                'Unable to find a converter for file %s', full_file_path)
        
        for recurse_sub_dir in recurse_sub_dirs:
            convert_files(root_dir, recurse_sub_dir, writer, recursive)