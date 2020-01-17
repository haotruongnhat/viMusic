from __future__ import division

import collections
import math
import copy

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib
from magenta.music import chord_symbols_lib
from magenta.music.protobuf import music_pb2
import tensorflow as tf

from .vimusic_velocity import *

from magenta.music.chord_symbols_lib import _CHORD_SYMBOL_REGEX

MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH

MAX_MIDI_VELOCITY = constants.MAX_MIDI_VELOCITY
MIN_MIDI_VELOCITY = constants.MIN_MIDI_VELOCITY
MAX_NUM_VELOCITY_BINS = MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1

STANDARD_PPQ = constants.STANDARD_PPQ

DEFAULT_MAX_SHIFT_STEPS = 100
DEFAULT_MAX_SHIFT_QUARTERS = 4

DEFAULT_PROGRAM = 0

STANDARD_PPQ = constants.STANDARD_PPQ
NOTES_PER_OCTAVE = constants.NOTES_PER_OCTAVE
NO_CHORD = constants.NO_CHORD

# Shortcut to CHORD_SYMBOL annotation type.
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL

_CHORD_SYMBOL_REGEX

def _program_and_is_drum_from_sequence(sequence, instrument=None):
	"""Get MIDI program and is_drum from sequence and (optional) instrument.

	Args:
		sequence: The NoteSequence from which MIDI program and is_drum will be
			extracted.
		instrument: The instrument in `sequence` from which MIDI program and
			is_drum will be extracted, or None to consider all instruments.

	Returns:
		A tuple containing program and is_drum for the sequence and optional
		instrument. If multiple programs are found (or if is_drum is True),
		program will be None. If multiple values of is_drum are found, is_drum
		will be None.
	"""
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

class ViMusicEvent(object):
	"""Class for storing events in a ViMusic."""

	# Start of a new note.
	NOTE_ON = 1
	# End of a note.
	NOTE_OFF = 2
	# Shift time forward.
	TIME_SHIFT = 3
	# Change current velocity.
	VELOCITY = 4
	#Start of new chord
	CHORD_ON = 5
	

	def __init__(self, event_type, event_value):
		if event_type in (ViMusicEvent.NOTE_ON, ViMusicEvent.NOTE_OFF):
			if not MIN_MIDI_PITCH <= event_value <= MAX_MIDI_PITCH:
				raise ValueError('Invalid pitch value: %s' % event_value)
		elif event_type == ViMusicEvent.TIME_SHIFT:
			if not 0 <= event_value:
				raise ValueError('Invalid time shift value: %s' % event_value)
		elif event_type == ViMusicEvent.VELOCITY:
			if not 1 <= event_value <= MAX_NUM_VELOCITY_BINS:
				raise ValueError('Invalid velocity value: %s' % event_value)
		elif event_type == ViMusicEvent.CHORD_ON:
			if not (_CHORD_SYMBOL_REGEX.match(event_value) or event_value == NO_CHORD):
				raise valueError('Invalid chord: {}'.format(event_value))
		else:
			raise ValueError('Invalid event type: %s' % event_type)

		self.event_type = event_type
		self.event_value = event_value

	def __repr__(self):
		return 'ViMusicEvent(%r, %r)' % (self.event_type, self.event_value)

	def __eq__(self, other):
		if not isinstance(other, ViMusicEvent):
			return False
		return (self.event_type == other.event_type and
			self.event_value == other.event_value)

class ViMusic(events_lib.EventSequence):
	def __init__(self, 
		quantized_sequence=None,steps_per_second=None,
		start_step=0, num_velocity_bins=0,
		max_shift_steps=DEFAULT_MAX_SHIFT_STEPS, 
		instrument=None,
		program=None, is_drum=None):
		""" 
		This is a type containing information about melodies, chords, and related stuff
		for the input of quantized sequence and instruments, it will extract information
		of chords and melodies depending on the instruments.
		The input of 
		"""
		if (quantized_sequence, steps_per_second).count(None) != 1:
			raise ValueError('Must specify exactly one of quantized_sequence or steps_per_second')

		if quantized_sequence:
			sequences_lib.assert_is_absolute_quantized_sequence(quantized_sequence)

			self._steps_per_second = (quantized_sequence.quantization_info.steps_per_second)

			#It is a tuple size-2 of list
			self._events = self._from_quantized_sequence(
				quantized_sequence, start_step, self._steps_per_second,num_velocity_bins,
				max_shift_steps=max_shift_steps, instrument=instrument)

			program, is_drum = _program_and_is_drum_from_sequence(
				quantized_sequence, instrument)

		else:
			self._steps_per_second = steps_per_second
			self._events = ([],[])
		
		if num_velocity_bins > MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1:
	  		raise ValueError('Number of velocity bins is too large: %d' % num_velocity_bins)

		self._start_step = start_step
		self._num_velocity_bins = num_velocity_bins
		self._max_shift_steps = max_shift_steps
		self._program = program
		self._is_drum = is_drum

	@staticmethod
	def _from_quantized_sequence(
		quantized_sequence, start_step, steps_per_second,
		num_velocity_bins, 
		max_shift_steps,
		instrument=None):
		"""
		Extract a list of events from the given quantized NoteSequence object,
		including the melodies and the chords
		"""
		current_step = start_step
		current_velocity_bin = 0
		vi_events = []
		#getting list of notes from the start step
		#and it is either have no instrument, or the instrument must match
		notes = [note for note in quantized_sequence.notes
		if note.quantized_start_step >= start_step
		and (instrument is None or note.instrument == instrument)]

		chords = [chord for chord in quantized_sequence.text_annotations
		if chord.annotation_type == CHORD_SYMBOL]
		chords = [chord for chord in chords
		if chord.quantized_step >= start_step]
		if len(chords) == 0: #make sure that there must be at least one chord
			annotation = quantized_sequence.text_annotations.add()
			annotation.time = start_step / steps_per_second
			annotation.quantized_step = start_step
			annotation.text = NO_CHORD
			annotation.annotation_type = CHORD_SYMBOL
			chords = [chord for chord in quantized_sequence.text_annotations
			if chord.annotation_type == CHORD_SYMBOL]
			chords = [chord for chord in chords
			if chord.quantized_step >= start_step]
		#sorting notes by start time, thenn with pitch
		sorted_notes = sorted(notes, key=lambda note: (note.start_time, note.pitch))
		sorted_chords = sorted(chords,key=lambda chord: chord.time)


		# Sort all note start and end events.
		onsets = [(note.quantized_start_step, idx, ViMusicEvent.NOTE_ON)
				for idx, note in enumerate(sorted_notes)]
		offsets = [(note.quantized_end_step, idx, ViMusicEvent.NOTE_OFF)
				for idx, note in enumerate(sorted_notes)]

		chord_events = sorted([(chord.quantized_step,idx,ViMusicEvent.CHORD_ON)
				for idx, chord in enumerate(sorted_chords)],key=lambda e : (e[0],e[1]))

		note_events = sorted(onsets + offsets)

		events = sorted(note_events + chord_events,
		key = lambda e : (e[0],e[1]))

		#Create TIME_SHIFT event for each shift
		#if the time takes for shifting is longer than max_shift_steps
		#then divide that into multiple steps
		for step, idx, status in events:
			if step > current_step:
				while step > current_step + max_shift_steps:
					vi_events.append(
						ViMusicEvent(event_type=ViMusicEvent.TIME_SHIFT,
						event_value=max_shift_steps)
					)
					current_step += max_shift_steps
				vi_events.append(
					ViMusicEvent(event_type=ViMusicEvent.TIME_SHIFT,
					event_value=int(step - current_step))
				)
				current_step = step

			if num_velocity_bins and status in (ViMusicEvent.NOTE_ON, ViMusicEvent.NOTE_OFF):
				velocity_bin = velocity_to_bin(
				sorted_notes[idx].velocity, num_velocity_bins)
				#if NOTE_ON and velocity changed
				if not status and velocity_bin != current_velocity_bin:
					#update new velocity
					current_velocity_bin = velocity_bin
					#add velocity change event
					vi_events.append(
					ViMusicEvent(event_type=ViMusicEvent.VELOCITY,
					event_value=current_velocity_bin))


			if status == ViMusicEvent.CHORD_ON:
				vi_events.append(
				ViMusicEvent(event_type=status,
				event_value=sorted_chords[idx].text))
			elif status in (ViMusicEvent.NOTE_ON,ViMusicEvent.NOTE_OFF):
				vi_events.append(
					ViMusicEvent(event_type=status,
					event_value=sorted_notes[idx].pitch)
				)
		return vi_events
	
	@property
	def steps_per_second(self):
		return self._steps_per_second

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
		"""Maximum step added is  bounded by max_shift_steps"""
		original_num_steps = num_steps
		if num_steps <= 0:
			return
		if (self._events and
			self._events[-1].event_type == ViMusicEvent.TIME_SHIFT):
			if self._events[-1].event_value < self._max_shift_steps:
			# Last event is already non-maximal time shift. Increase its duration.
				added_steps = min(num_steps,
							self._max_shift_steps - self._events[-1].event_value)
				self._events[-1] = ViMusicEvent(
				ViMusicEvent.TIME_SHIFT,
				self._events[-1].event_value + added_steps)
				num_steps -= added_steps

		while num_steps >= self._max_shift_steps:
			self._events.append(
			ViMusicEvent(event_type=ViMusicEvent.TIME_SHIFT,
			event_value=self._max_shift_steps))
			num_steps -= self._max_shift_steps

		if num_steps > 0:
			self._events.append(
			ViMusicEvent(event_type=ViMusicEvent.TIME_SHIFT,
							event_value=num_steps))

	def _trim_steps(self, num_steps):
		"""Trims a given number of steps from the end of the sequence."""
		original_num_steps = num_steps
		steps_trimmed = 0
		num_steps = copy.deepcopy(original_num_steps)
		while self._events and steps_trimmed < num_steps:
			if self._events[-1].event_type == ViMusicEvent.TIME_SHIFT:
				if steps_trimmed + self._events[-1].event_value > num_steps:
					self._events[-1] = ViMusicEvent(
					event_type=ViMusicEvent.TIME_SHIFT,
					event_value=(self._events[-1].event_value -
					num_steps + steps_trimmed))
					steps_trimmed = num_steps
				else:
					steps_trimmed += self._events[-1].event_value
					self._events.pop()
			else:
				self._events.pop()

	def set_length(self, steps, from_left=False):
		if from_left:
			raise NotImplementedError('from_left is not supported')
		if self.num_steps < steps:
			self._append_steps(steps - self.num_steps)
		elif self.num_steps > steps:
			self._trim_steps(self.num_steps - steps)
	
	def append(self, event):
		"""Appends the event to the end of the sequence.

		Args:
		event: The performance event to append to the end.

		Raises:
		ValueError: If `event` is not a valid performance event.
		"""
		if not isinstance(event, ViMusicEvent):
			raise ValueError('Invalid vimusic event: %s' % event)
		self._events.append(event)
		

	def truncate(self, num_events):
		"""Truncates this Performance to the specified number of events.

		Args:
		num_events: The number of events to which this performance will be
			truncated.
		"""
		self._events = self._events[:num_events]

	def __len__(self):
		#from pdb import set_trace ; set_trace()
		return len(self._events)
		
	def __getitem__(self, i):
		"""Returns the event at the given index."""
		return self._events[i]
		
	def __iter__(self):
		"""Return an iterator over the events in this sequence."""
		return iter(self._events)

	def __str__(self):
		strs = []
		for event in self._events:
			if event.event_type == ViMusicEvent.NOTE_ON:
				strs.append('(%s, ON)' % event.event_value)
			elif event.event_type == ViMusicEvent.NOTE_OFF:
				strs.append('(%s, OFF)' % event.event_value)
			elif event.event_type == ViMusicEvent.TIME_SHIFT:
				strs.append('(%s, TIME_SHIFT)' % event.event_value)
			elif event.event_type == ViMusicEvent.VELOCITY:
				strs.append('(%s, VELOCITY)' % event.event_value)
			elif event.event_type == ViMusicEvent.TIME_SHIFT:
				strs.append('(%s, TIME_SHIFT)' % event.event_value)
			elif event.event_type == ViMusicEvent.CHORD_ON:
				strs.append('(%s, CHORD_ON)' % event.event_value)
			else:
				raise ValueError('Unknown event type: %s' % event.event_type)
		return str(strs)

	@property
	def end_step(self):
		return self.start_step + self.num_steps

	@property
	def num_steps(self):
		steps = 0
		for event in self:
			if event.event_type == ViMusicEvent.TIME_SHIFT:
				steps += event.event_value
		return steps

	def to_sequence(self,
		velocity=100,
		instrument=0,
		program=None,
		max_note_duration=None,
		render_chord_from_annotation=False):
		seconds_per_step = 1.0 / self.steps_per_second
		return self._to_sequence(
		seconds_per_step=seconds_per_step,
		velocity=velocity,
		instrument=instrument,
		program=program,
		max_note_duration=max_note_duration,
		render_chord_from_annotation=render_chord_from_annotation)

	def _to_sequence(self, seconds_per_step, velocity, instrument, program,
		max_note_duration=None,render_chord_from_annotation=False):

		steps_per_second = 1 / seconds_per_step
		sequence_start_time = self.start_step * seconds_per_step

		sequence = music_pb2.NoteSequence()
		sequence.ticks_per_quarter = STANDARD_PPQ

		step = 0

		if program is None:
			# Use program associated with the performance (or default program).
			program = self.program if self.program is not None else DEFAULT_PROGRAM
		is_drum = self.is_drum if self.is_drum is not None else False

		# Map pitch to list because one pitch may be active multiple times.
		#initialize dict, but default type of value is list
		pitch_start_steps_and_velocities = collections.defaultdict(list)
		chord_set = set()
		for i, event in enumerate(self):
			if event.event_type == ViMusicEvent.NOTE_ON:
				pitch_start_steps_and_velocities[event.event_value].append(
				(step, velocity))
			elif event.event_type == ViMusicEvent.NOTE_OFF:
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
			elif event.event_type == ViMusicEvent.TIME_SHIFT:
				step += event.event_value
			elif event.event_type == ViMusicEvent.VELOCITY:
				assert self._num_velocity_bins
				velocity = velocity_bin_to_velocity(
					event.event_value, self._num_velocity_bins)
			elif event.event_type == ViMusicEvent.CHORD_ON:
				chord_set.add((event.event_value,step))


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

		if not render_chord_from_annotation:
			for chord in chord_set:
				annotation = sequence.text_annotations.add()
				annotation.time = chord[1] / steps_per_second
				annotation.quantized_step = chord[1]
				annotation.text = chord[0]
				annotation.annotation_type = CHORD_SYMBOL
		else:
			chord_set = sorted(chord_set, lambda x: x[1])
			chord_set.append(("N.C",step))
			for i, chord in enumerate(chord_set[:-1]):
				pitches = chord_symbols_lib.chord_symbol_pitches(chord[0])
				start_quantized_step = chord[1]
				end_quantized_step = chord_set[i + 1][1]
				start_time = start_quantized_step / steps_per_second
				end_time = end_quantized_step / steps_per_second
				for pitch in pitches: 
					pass


		return sequence
