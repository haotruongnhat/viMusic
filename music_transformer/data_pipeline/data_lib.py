from __future__ import division

import abc
import collections
import math
import os
import copy
import syllables

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib
from music_transformer.music import vimusic_pb2
from magenta.music import note_sequence_io
from magenta.pipelines import pipeline
import tensorflow as tf

from music_transformer.utils.constants import *

from music_transformer.utils import default_config

import warnings


def _velocity_bin_size(num_velocity_bins):
    return int(math.ceil(
    (MAX_VELOCITY - MIN_VELOCITY + 1) / num_velocity_bins))


def velocity_to_bin(velocity, num_velocity_bins):
    return ((velocity - MIN_VELOCITY) //
    _velocity_bin_size(num_velocity_bins) + 1)


def velocity_bin_to_velocity(velocity_bin, num_velocity_bins):
    return (
    MIN_VELOCITY + (velocity_bin - 1) *
    _velocity_bin_size(num_velocity_bins))


def _program_and_is_drum_from_sequence(sequence, instrument=None):
    """
    Check if there is drum in the sequence
    Retrieve a set of program used in the sequence
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

def unify_beat_in_sequence(sequence,bpm):
    unify_sequence = copy.deepcopy(sequence)
    if not unify_sequence.tempos:
        tempo = unify_sequence.tempos.add()
        tempo.qpm = bpm
        tempo.time = 0
    tempos = copy.deepcopy(sorted(unify_sequence.tempos, key=lambda t: t.time))
    for note in unify_sequence.notes:
        if len(tempos) > 1:
            if note.start_time >= tempos[1].time:
                del tempos[0]
        duration_in_time = note.end_time - note.start_time
        duration_for_half = bpm / tempos[0].qpm
        if duration_in_time > 0:
            num_of_half = duration_in_time / duration_for_half
            if num_of_half < 1:
                num_denominator = int(math.ceil(2 / num_of_half))
                loss_inverse = num_denominator % 4
                add_in_inverse = num_denominator + loss_inverse
                note.numerator = 1
                note.denominator = add_in_inverse
            else:
                num_of_half = int(math.ceil(num_of_half))
                if num_of_half >= 2:
                    loss_half = num_of_half % 2
                    add_in_half = num_of_half + loss_half
                    note.numerator = int(add_in_half / 2)
                    note.denominator = 1
                else:
                    note.numerator = num_of_half
                    note.denominator = 2
        else:
            raise NegativeTimeError("Beats should not be negative")
    """ For testing only
    with open("d.txt","w") as ff:
        for s1,s2 in zip(sequence.notes,unify_sequence.notes):
            ff.write("{} {} {} {}\n".format(s1.numerator,s2.numerator,s1.denominator,s2.denominator))
    """
    while 0 < len(unify_sequence.tempos):
        del unify_sequence.tempos[0]
    tempo = unify_sequence.tempos.add()
    tempo.qpm = DEFAULT_BPM
    tempo.time = 0
    return unify_sequence

class ViMusicEvent(object):

    NEW_NOTE_NO_SYLLABLE = 0
    NEW_NOTE_ONE_SYLLABLE = 1
    NEW_NOTE_TWO_SYLLABLE = 2
    NEW_NOTE_THREE_SYLLABLE = 3
    NEW_NOTE_FOUR_SYLLABLE = 4
    END_NOTE = 5
    STEP_SHIFT = 6
    VELOCITY_CHANGE = 7
    def __init__(self,event_type,event_value):
        if event_type < ViMusicEvent.NEW_NOTE_NO_SYLLABLE or event_type > ViMusicEvent.VELOCITY_CHANGE:
            raise ValueError("Invalid event")

        self.event_type = event_type
        self.event_value = event_value

    def __repr__(self):
        return 'ViMusicEvent({},{})'.format(self.event_type,self.event_value)

    def __eq__(self, other):
        if not isinstance(other, ViMusicEvent):
            return False
        return (self.event_type == self.event_type and
        self.event_value == self.event_value)

class ViMusic(events_lib.EventSequence):
    def __init__(self,quantized_sequence,
    start_step, num_velocity_bins,instrument=None):
        """Create a ViMusic sequence consisting of ViMusicEvents"""
        #Check if the values are satisfied
        if num_velocity_bins > MAX_VELOCITY - MIN_VELOCITY + 1:
            raise ValueError('Number of velocity bins is too large: %d' % num_velocity_bins)

        program, is_drum = _program_and_is_drum_from_sequence(
        quantized_sequence, instrument)
        
        self._start_step = start_step
        self._num_velocity_bins = num_velocity_bins
        self._max_shift_steps = DEFAULT_MAX_SHIFT_STEPS
        self._program = program
        self._is_drum = is_drum

        #Constructing ViMusic from events
        self._events = self._from_quantized_sequence(
        quantized_sequence,start_step,num_velocity_bins,
        self._max_shift_steps,instrument)

    @property
    def start_step(self):
        return self._start_step

    @property
    def program(self):
        return self._program

    @property
    def is_drum(self):
        return self._is_drum

    def _append_steps(self, num_steps):
        if (self._events and
        self._events[-1].event_type == ViMusicEvent.STEP_SHIFT and
        self._events[-1].event_value < self._max_shift_steps):
            added_steps = min(num_steps,
            self._max_shift_steps - self._events[-1].event_value)
            self._events[-1] = ViMusicEvent(
                ViMusicEvent.STEP_SHIFT,
                self._events[-1].event_value + added_steps
            )
            num_steps -= added_steps

        while num_steps >= self._max_shift_steps:
            self._events.append(
                ViMusicEvent(ViMusicEvent.STEP_SHIFT,
                self._max_shift_steps)
            )
            num_steps -= self._max_shift_steps

        if num_steps > 0:
            self._events.append(
                ViMusicEvent(ViMusicEvent.STEP_SHIFT,
                num_steps)
            )

    def _trim_steps(self, num_steps):
        steps_trimmed = 0
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

        assert self.num_steps == steps

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
        return  "\n".join(str(x) for x in self._events)

    @property
    def end_step(self):
        return self._start_step + self.num_steps

    @property
    def num_steps(self):
        """
        Length will equal to the shortest first beat of the event with 
        the sum of other rest note afterward
        """
        steps = 0
        for event in self:
            if event.event_type == ViMusicEvent.STEP_SHIFT:
                steps += event.event_value
        return steps

    @property
    def steps(self):
        step = self._start_step
        result = []
        for event in self:
            result.append(step)
            if event.event_type == ViMusicEvent.STEP_SHIFT:
                step += event.event_value
        return result

    @staticmethod
    def _from_quantized_sequence(quantized_sequence, start_step,
    num_velocity_bins,max_shift_steps,instrument=None):
        
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

        #Change text to the highest notes
        current_step = start_step
        group_note_idx = []
        
        for step, idx, _ in onsets:
            if current_step != step:
                if len(group_note_idx) > 1:
                    #update text for the highest note:
                    target = [(idx,sorted_notes[idx].pitch,sorted_notes[idx].text) 
                    for idx in group_note_idx]
                    target = sorted(filter(lambda x: x[2] is not None,target),reverse=True,
                    key=lambda x : (x[0],x[1]))
                    if len(target) > 1:
                        sorted_notes[target[0][0]].text = target[0][2]
                group_note_idx = []
                current_step = step

            if current_step == step:
                if sorted_notes[idx].text is not None:
                    group_note_idx.append(idx)

        current_step = start_step
        current_velocity_bin = 0
        events = []
        for step, idx, is_offset in note_events:
            #Do the shifting task
            if step > current_step:
                #Do the shifting task
                while step > current_step + max_shift_steps:
                    # We need to move further than the maximum shift size.
                    events.append(
                        ViMusicEvent(event_type=ViMusicEvent.STEP_SHIFT,
                               event_value=max_shift_steps))
                    current_step += max_shift_steps
                events.append(
                ViMusicEvent(event_type=ViMusicEvent.STEP_SHIFT,
                             event_value=int(step - current_step)))
                current_step = step

            #Do the velocity task
            if num_velocity_bins:
                velocity_bin = velocity_to_bin(
                sorted_notes[idx].velocity, num_velocity_bins)
                if not is_offset and velocity_bin != current_velocity_bin:
                    current_velocity_bin = velocity_bin
                    events.append(
                    ViMusicEvent(event_type=ViMusicEvent.VELOCITY_CHANGE,
                    event_value=current_velocity_bin))

            #Do for the new note
            if not is_offset:
                #get syllable
                if sorted_notes[idx].text is not None:
                    new_node_code = syllables.estimate(sorted_notes[idx].text)
                    if new_node_code > ViMusicEvent.NEW_NOTE_FOUR_SYLLABLE:
                        new_node_code = ViMusicEvent.NEW_NOTE_FOUR_SYLLABLE

                else:
                    new_node_code = ViMusicEvent.NEW_NOTE_NO_SYLLABLE
                events.append(
                ViMusicEvent(event_type=new_node_code,
                event_value=sorted_notes[idx].pitch))

            #Do for the end note
            if is_offset:
                events.append(
                ViMusicEvent(event_type=ViMusicEvent.END_NOTE,
                            event_value=sorted_notes[idx].pitch))

        return events

    '''
    TODO: Incorporate lyrics to the generate sequence
    def to_sequence(self, steps_per_second, velocity, instrument, program,
                   max_note_duration=None):
        seconds_per_step = 1.0 / steps_per_second
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
            if event.event_type == ViMusicEvent.:
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