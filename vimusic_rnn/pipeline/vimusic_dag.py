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

"""Pipeline to create Performance dataset."""
from magenta.music.protobuf import music_pb2
from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipelines_common
from magenta.pipelines import lead_sheet_pipelines

from magenta.pipelines.performance_pipeline import PerformanceExtractor

from .vimusic_pipeline import ViMusicEncoderDecoderPipeline
from .vimusic_pipeline import ViMusicQuantizer
from .vimusic_pipeline import ViMusicGetter
#Leadsheet -> (use ViMusicPipeline) -> MyType -> (use EventSequencePipeline) -> tf.SequenceExample
#dag pipeline to preprocess data
def get_pipeline(config, min_events, max_events, eval_ratio):
    """Returns the Pipeline instance which creates the RNN dataset.

    Args:
    config: A PerformanceRnnConfig.
    min_events: Minimum number of events for an extracted sequence.
    max_events: Maximum number of events for an extracted sequence.
    eval_ratio: Fraction of input to set aside for evaluation set.

    Returns:
    A pipeline.Pipeline instance.
    """
    all_transpositions = config.transpose_to_key is None
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    stretch_factors = [0.95, 0.975, 1.0, 1.025, 1.05]

    # Transpose no more than a major third.
    transposition_range = range(-3, 4)

    partitioner = pipelines_common.RandomPartition(
        music_pb2.NoteSequence,
        ['eval_vimusic', 'train_vimusic'],
        [eval_ratio])
    dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

    for mode in ['eval', 'train']:
        #1. sustain note sequence
        sustain_pipeline = note_sequence_pipelines.SustainPipeline(
        name='SustainPipeline_' + mode)

        #2. stretch pipeline
        stretch_pipeline = note_sequence_pipelines.StretchPipeline(
        stretch_factors if mode == 'train' else [1.0],
        name='StretchPipeline_' + mode)

        #3x. splitter (for performance)
        splitter = note_sequence_pipelines.Splitter(
        hop_size_seconds=30.0, name='Splitter_' + mode)

        #3. split based on time signature
        time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)

        #4. quantizer
        quantizer = ViMusicQuantizer(steps_per_quarter=config.steps_per_quarter,
            steps_per_second=config.steps_per_second,name='Quantizer_' + mode)

        #5. lead sheet extractor
        lead_sheet_extractor = lead_sheet_pipelines.LeadSheetExtractor(
        min_bars=7, max_steps=512, min_unique_pitches=3, gap_bars=1.0,
        ignore_polyphonic_notes=False, all_transpositions=all_transpositions,
        name='LeadSheetExtractor_' + mode)

        #5. transposition pipeline
        transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range if mode == 'training' else [0],
        name='TranspositionPipeline_' + mode)

        #6. performance extractor
        perf_extractor = PerformanceExtractor(
        min_events=min_events, max_events=max_events,
        num_velocity_bins=config.num_velocity_bins,
        note_performance=config.note_performance,
        name='PerformanceExtractor_' + mode)

        #7. vimusic encoder decoder that incorporates data from improv and performance
        encoder_pipeline = ViMusicEncoderDecoderPipeline(config, name='EventSequencePipeline_' + mode)

        #8. for getting notesequence
        improv_getter_after_quantized = ViMusicGetter("improv",name="GetterPipeline_improv_" + mode)
        perf_getter_after_quantized = ViMusicGetter("perf",name="GetterPipeline_perf_" + mode)

        #8. dag pipeline construction

        #8a. performance dag pipeline
        dag[sustain_pipeline] = partitioner[mode + '_vimusic']
        dag[stretch_pipeline] = sustain_pipeline
        dag[splitter] = stretch_pipeline
        dag[quantizer] = {"improv": time_change_splitter, "perf": splitter}
        dag[perf_getter_after_quantized] = quantizer
        dag[transposition_pipeline] = perf_getter_after_quantized
        dag[perf_extractor] = transposition_pipeline

        #8b. improv rnn pipeline
        dag[time_change_splitter] = partitioner[mode + '_vimusic']
        dag[improv_getter_after_quantized] = quantizer
        dag[lead_sheet_extractor] = improv_getter_after_quantized

        #8c. final encoding
        dag[encoder_pipeline] = {"improv": lead_sheet_extractor, "perf": perf_extractor}

        dag[dag_pipeline.DagOutput(mode + '_vimusic')] = encoder_pipeline

    return dag_pipeline.DAGPipeline(dag)