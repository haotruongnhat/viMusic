# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Melody RNN model."""

import copy

import magenta
from magenta.models.shared import events_rnn_model
import magenta.music as mm
from magenta.models.vivi_rnn import vivi_encoder_decoder
from tensorflow.contrib import training as contrib_training

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = None


class ViviRnnModel(events_rnn_model.EventSequenceRnnModel):
  """Class for RNN melody-given-chords generation models."""

  def generate_polyphonic_sequence( self,
                                    num_steps, primer_sequence, backing_chords,
                                    temperature=1.0, beam_size=1,
                                    branch_factor=1, steps_per_iteration=1,
                                    modify_events_callback=None):
    """
    Generate a polyphonic track from a primer polyphonic track
    and backing chords

    Args:
      num_steps: The integer length in steps of the final track, after
          generation. Includes the primer.
      primer_sequence: The primer sequence, a PolyphonicSequence object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes tracks more
         random, less than 1.0 makes tracks less random.
      beam_size: An integer, beam size to use when generating tracks via
          beam search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of steps to take per beam search
          iteration.
      modify_events_callback: An optional callback for modifying the event list.
          Can be used to inject events rather than having them generated. If not
          None, will be called with 3 arguments after every event: the current
          EventSequenceEncoderDecoder, a list of current EventSequences, and a
          list of current encoded event inputs.
    Returns:
      The generated PolyphonicSequence object (which begins with the provided
      primer track).
    """
    num_steps = len(backing_chords)
    return self._generate_events(num_steps, primer_sequence, temperature,
                              beam_size, branch_factor, steps_per_iteration,
                              modify_events_callback=modify_events_callback,
                              control_events=backing_chords)


  def generate_melody(self, primer_melody, backing_chords, temperature=1.0,
                      beam_size=1, branch_factor=1, steps_per_iteration=1):
    """Generate a melody from a primer melody and backing chords.

    Args:
      primer_melody: The primer melody, a Melody object. Should be the same
          length as the primer chords.
      backing_chords: The backing chords, a ChordProgression object. Must be at
          least as long as the primer melody. The melody will be extended to
          match the length of the backing chords.
      temperature: A float specifying how much to divide the logits by
          before computing the softmax. Greater than 1.0 makes melodies more
          random, less than 1.0 makes melodies less random.
      beam_size: An integer, beam size to use when generating melodies via beam
          search.
      branch_factor: An integer, beam search branch factor to use.
      steps_per_iteration: An integer, number of melody steps to take per beam
          search iteration.

    Returns:
      The generated Melody object (which begins with the provided primer
          melody).
    """
    melody = copy.deepcopy(primer_melody)
    chords = copy.deepcopy(backing_chords)

    transpose_amount = melody.squash(
        self._config.min_note,
        self._config.max_note,
        self._config.transpose_to_key)
    chords.transpose(transpose_amount)

    num_steps = len(chords)
    melody = self._generate_events(num_steps, melody, temperature, beam_size,
                                   branch_factor, steps_per_iteration,
                                   control_events=chords)

    melody.transpose(-transpose_amount)

    return melody

  def melody_log_likelihood(self, melody, backing_chords):
    """Evaluate the log likelihood of a melody conditioned on backing chords.

    Args:
      melody: The Melody object for which to evaluate the log likelihood.
      backing_chords: The backing chords, a ChordProgression object.

    Returns:
      The log likelihood of `melody` conditioned on `backing_chords` under this
      model.
    """
    melody_copy = copy.deepcopy(melody)
    chords_copy = copy.deepcopy(backing_chords)

    transpose_amount = melody_copy.squash(
        self._config.min_note,
        self._config.max_note,
        self._config.transpose_to_key)
    chords_copy.transpose(transpose_amount)

    return self._evaluate_log_likelihood([melody_copy],
                                         control_events=chords_copy)[0]


class ViviRnnConfig(events_rnn_model.EventSequenceRnnConfig):
  """Stores a configuration for an ViviRnn.

  You can change `min_note` and `max_note` to increase/decrease the melody
  range. Since melodies are transposed into this range to be run through
  the model and then transposed back into their original range after the
  melodies have been extended, the location of the range is somewhat
  arbitrary, but the size of the range determines the possible size of the
  generated melodies range. `transpose_to_key` should be set to the key
  that if melodies were transposed into that key, they would best sit
  between `min_note` and `max_note` with having as few notes outside that
  range. If `transpose_to_key` is None, melodies and chords will not be
  transposed at generation time, but all of the training data will be transposed
  into all 12 keys.

  Attributes:
    details: The GeneratorDetails message describing the config.
    encoder_decoder: The EventSequenceEncoderDecoder object to use.
    hparams: The HParams containing hyperparameters to use.
    min_note: The minimum midi pitch the encoded melodies can have.
    max_note: The maximum midi pitch (exclusive) the encoded melodies can have.
    transpose_to_key: The key that encoded melodies and chords will be
        transposed into, or None if they should not be transposed. If None, all
        of the training data will be transposed into all 12 keys.
  """

  def __init__(self, details, encoder_decoder, hparams,
               min_note=DEFAULT_MIN_NOTE, max_note=DEFAULT_MAX_NOTE,
               transpose_to_key=DEFAULT_TRANSPOSE_TO_KEY):
    super(ViviRnnConfig, self).__init__(details, encoder_decoder, hparams)

    if min_note < mm.MIN_MIDI_PITCH:
      raise ValueError('min_note must be >= 0. min_note is %d.' % min_note)
    if max_note > mm.MAX_MIDI_PITCH + 1:
      raise ValueError('max_note must be <= 128. max_note is %d.' % max_note)
    if max_note - min_note < mm.NOTES_PER_OCTAVE:
      raise ValueError('max_note - min_note must be >= 12. min_note is %d. '
                       'max_note is %d. max_note - min_note is %d.' %
                       (min_note, max_note, max_note - min_note))
    if (transpose_to_key is not None and
        (transpose_to_key < 0 or transpose_to_key > mm.NOTES_PER_OCTAVE - 1)):
      raise ValueError('transpose_to_key must be >= 0 and <= 11. '
                       'transpose_to_key is %d.' % transpose_to_key)

    self.min_note = min_note
    self.max_note = max_note
    self.transpose_to_key = transpose_to_key


# Default configurations.
default_configs = {
    'vivi':
        ViviRnnConfig(
            magenta.music.protobuf.generator_pb2.GeneratorDetails(
                id='vivi',
                description='Polyphony-given-chords RNN with chord pitches encoding.'
            ),
            magenta.music.ConditionalEventSequenceEncoderDecoder(
                magenta.music.PitchChordsEncoderDecoder(),
                magenta.music.OneHotEventSequenceEncoderDecoder(
                    vivi_encoder_decoder.PolyphonyOneHotEncoding())),
            contrib_training.HParams(
                batch_size=128,
                rnn_layer_sizes=[256, 256, 256],
                dropout_keep_prob=0.5,
                clip_norm=3,
                learning_rate=0.001))
}
