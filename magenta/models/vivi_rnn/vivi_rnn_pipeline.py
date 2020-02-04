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

"""Pipeline to create ViviRNN dataset."""

import magenta
from magenta.music.protobuf import music_pb2
from magenta.pipelines import dag_pipeline
from magenta.pipelines import lead_sheet_pipelines
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import statistics
from magenta.music import chord_symbols_lib
from magenta.music import chords_lib
from magenta.music import events_lib
from magenta.music import lead_sheets_lib
from magenta.music import LeadSheet
from magenta.music import sequences_lib
from magenta.pipelines import chord_pipelines
from magenta.pipelines import melody_pipelines
from magenta.models.polyphony_rnn import polyphony_lib
from magenta.models.improv_rnn import improv_rnn_pipeline

from magenta.models.vivi_rnn import vivi_lib
import tensorflow as tf
import copy

class PolyphonicLeadSheetExtractor(pipeline.Pipeline):
  """Extracts polyphonic lead sheet fragments from a quantized NoteSequence."""

  def __init__(self, min_bars=7, max_steps=512, min_unique_pitches=5,
               gap_bars=1.0, name=None):
    super(PolyphonicLeadSheetExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=vivi_lib.PolyphonicLeadSheet,
        name=name)
    self._min_bars = min_bars
    self._max_steps = max_steps
    self._min_unique_pitches = min_unique_pitches
    self._gap_bars = gap_bars

  def transform(self, quantized_sequence):
    try:
      lead_sheets, stats = extract_polyphonic_lead_sheet_fragments(quantized_sequence)
    except events_lib.NonIntegerStepsPerBarError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      lead_sheets = []
      stats = [statistics.Counter('non_integer_steps_per_bar', 1)]
    except chord_symbols_lib.ChordSymbolError as detail:
      tf.logging.warning('Skipped sequence: %s', detail)
      lead_sheets = []
      stats = [statistics.Counter('chord_symbol_exception', 1)]
    self._set_stats(stats)
    return lead_sheets

def extract_polyphonic_lead_sheet_fragments(quantized_sequence,
                                            min_steps=0, max_steps=100):
  sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
  stats = dict([('empty_chord_progressions',
                 statistics.Counter('empty_chord_progressions'))])

  melodies, melody_stats = polyphony_lib.extract_polyphonic_sequences(quantized_sequence,
                                                                      min_steps_discard=min_steps,
                                                                      max_steps_discard=max_steps)
  chord_progressions, chord_stats = chord_pipelines.extract_chords_for_melodies(quantized_sequence, 
                                                                                melodies)
  lead_sheets = []
  for melody, chords in zip(melodies, chord_progressions):
    # If `chords` is None, it's because a chord progression could not be
    # extracted for this particular melody.
    if chords is not None:
      if all(chord == chords_lib.NO_CHORD for chord in chords):
        stats['empty_chord_progressions'].increment()
      else:
        lead_sheet = vivi_lib.PolyphonicLeadSheet(melody, chords)
        if True:#all_transpositions:
          for amount in range(-6, 6):
            transposed_lead_sheet = copy.deepcopy(lead_sheet)
            transposed_lead_sheet.transpose(amount)
            lead_sheets.append(transposed_lead_sheet)
        else:
          lead_sheets.append(lead_sheet)
  return lead_sheets, list(stats.values()) + melody_stats + chord_stats

class EncoderPipeline(pipeline.Pipeline):
  """A Module that converts lead sheets to a model specific encoding."""

  def __init__(self, config, name):
    """Constructs an EncoderPipeline.

    Args:
      config: An ViviRnnConfig that specifies the encoder/decoder,
          pitch range, and transposition behavior.
      name: A unique pipeline name.
    """
    super(EncoderPipeline, self).__init__(
        input_type=magenta.music.LeadSheet,
        output_type=tf.train.SequenceExample,
        name=name)
    self._conditional_encoder_decoder = config.encoder_decoder
    self._min_note = config.min_note
    self._max_note = config.max_note
    self._transpose_to_key = config.transpose_to_key

  def transform(self, lead_sheet):
    lead_sheet.squash(
        self._min_note,
        self._max_note,
        self._transpose_to_key)
    try:
      encoded = [self._conditional_encoder_decoder.encode(
          lead_sheet.chords, lead_sheet.melody)]
      stats = []
    except magenta.music.ChordEncodingError as e:
      tf.logging.warning('Skipped lead sheet: %s', e)
      encoded = []
      stats = [statistics.Counter('chord_encoding_exception', 1)]
    except magenta.music.ChordSymbolError as e:
      tf.logging.warning('Skipped lead sheet: %s', e)
      encoded = []
      stats = [statistics.Counter('chord_symbol_exception', 1)]
    self._set_stats(stats)
    return encoded

  def get_stats(self):
    return {}

def get_pipeline(config, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: An ViviRnnConfig object.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  all_transpositions = config.transpose_to_key is None
  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_lead_sheets', 'training_lead_sheets'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_quarter=config.steps_per_quarter, name='Quantizer_' + mode)
    chord_infer = improv_rnn_pipeline.InferChordFromQuantizedSequence(name='ChordInfer_' + mode)
    polyphonic_lead_sheet_extractor = PolyphonicLeadSheetExtractor(
        min_bars=7, max_steps=512, min_unique_pitches=3, gap_bars=1.0,
        name='PolyphonicLeadSheetExtractor_' + mode)
    encoder_pipeline = EncoderPipeline(config, name='EncoderPipeline_' + mode)

    dag[time_change_splitter] = partitioner[mode + 'polyphonic_lead_sheets']
    dag[quantizer] = time_change_splitter
    dag[chord_infer] = quantizer
    dag[polyphonic_lead_sheet_extractor] = chord_infer
    dag[encoder_pipeline] = polyphonic_lead_sheet_extractor
    dag[dag_pipeline.DagOutput(mode + 'polyphonic_lead_sheets')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)
