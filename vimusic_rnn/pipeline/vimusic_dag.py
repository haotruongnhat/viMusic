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

from .vimusic_pipeline import ViMusicEncoderDecoderPipeline
from .vimusic_pipeline import ViMusicExtractor
from .vimusic_pipeline import InferChordsPipeline
#Leadsheet -> (use ViMusicPipeline) -> MyType -> (use EventSequencePipeline) -> tf.SequenceExample
#dag pipeline to preprocess data
#TODO: Create a pipeline that incorporate imrpov rnn and performance rnn
#Some differences between these 2 models
#how they quantize the note sequences
#how they split the note sequences
#how they extract information
#LeadSheet extractor wants their information to be quantized based steps_per_quarter
#Performance Extractor do not care
def get_pipeline(config, eval_ratio):
    """Returns the Pipeline instance which creates the RNN dataset.

    Args:
    config: A PerformanceRnnConfig.
    min_events: Minimum number of events for an extracted sequence.
    max_events: Maximum number of events for an extracted sequence.
    eval_ratio: Fraction of input to set aside for evaluation set.

    Returns:
    A pipeline.Pipeline instance.
    """
    # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
    stretch_factors = [0.95, 0.975, 1.0, 1.025, 1.05]

    # Transpose no more than a major third.
    transposition_range = range(-3, 4)

    #split note sequence datasets into multiple data sets
    partitioner = pipelines_common.RandomPartition(
        music_pb2.NoteSequence,
        ['eval_vimusic', 'train_vimusic'],
        [eval_ratio])
    dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

    #for each datasets
    for mode in ['eval', 'train']:
        #1. sustain note sequence
        #why the hell do we need to sustain notes ?????
        """
        Apply lengthening the end time to the upcoming control change time
        Currently this is used for pedal changes of the piano (default: 64)
        This is not applicable for drums.
        Why do we need to lengthen these notes based on control changes? I don't get it :v
        """
        sustain_pipeline = note_sequence_pipelines.SustainPipeline(
        name='SustainPipeline_' + mode)

        #2. stretch pipeline
        """
        Only unquantized notes can be stretched
        For train data, one NoteSequence instance will be stretched into
        multiple values based on stretch factors
        value larger than 1 will create slow notesequence, and vice versa
        it is just about creating multiple version of note sequence,
        each having different speed yo :v
        """
        stretch_pipeline = note_sequence_pipelines.StretchPipeline(
        stretch_factors if mode == 'train' else [1.0],
        name='StretchPipeline_' + mode)

        #3x. splitter (for performance)
        """
        hop_size_seconds is a scalar -> np.arange(hop_size_seconds, total_time, hop_size_seconds)
        hop_size_seconds is a list -> list based on list
        Splitter simply split a note sequence into a list of note sequences,
        where each note sequences has a total_time of hop_size_seconds
        """
        splitter = note_sequence_pipelines.Splitter(
        hop_size_seconds=30.0, name='Splitter_' + mode)
        """
        just like Splitter, but this split note sequences into a list of note sequences,
        each bassed on different time change (time signature and tempos changes).
        If there is n time change on different time, split into n note sequences
        If some of the time changes are similar and continous, these will not be splitted
        """
        #3. split based on time signature
        """
        time_change_splitter_improv = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_improv_' + mode)
        time_change_splitter_perf = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_perf_' + mode)
        """

        #4. quantizer
        """
        There are two versions of quantizer

        quantize_note_sequence: this based on steps_per_quarter and qpm
        It caculates the quantized_start_step and quantized_end_step of each note,
        based on its start_time and end_time
        e.g: steps_per_quarter: 4, qpm: 60 (qps: 1), start_time: 2.5, end_time: 3.5
        quantized_start_step = (2.5 - 0) * qps * steps_per_quarter = 10 (floor)
        quantized_end_step = (3.5 - 0) * qps * steps_per_quarter = 14 (floor)

        quantize_note_sequence_absolute: this ignore the qpm of note sequence
        and calculate using this formula
        quantized_start/end_step = (start/end_time - 0.0) * steps_per_seconds
        """
        quantizer = note_sequence_pipelines.Quantizer(
        steps_per_second=config.steps_per_second, name='Quantizer_' + mode)

        infer_chords = InferChordsPipeline(
            name = 'InferChord_' + mode
        )
        #5. transposition pipeline
        """
        This transposition only change the pitches of the notes
        time signatures, chord, pitch_names of notes is ignored
        from a note sequence, it will create a list of  new note sequences,
        each is transposed by each value of transposition_range
        """
        transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range if mode == 'training' else [0],
        name='TranspositionPipeline_' + mode)

        #6. performance extractor
        vi_extractor = ViMusicExtractor(
        config,name='ViMusicExtractor_' + mode)

        #7. vimusic encoder decoder that incorporates data from improv and performance
        encoder_pipeline = ViMusicEncoderDecoderPipeline(
        config, name='EventSequencePipeline_' + mode)

        #8. dag pipeline construction
        dag[sustain_pipeline] = partitioner[mode + '_vimusic']
        dag[stretch_pipeline] = sustain_pipeline
        dag[splitter] = stretch_pipeline
        dag[quantizer] = splitter
        dag[infer_chords] = quantizer
        dag[transposition_pipeline] = infer_chords
        dag[vi_extractor] = transposition_pipeline
        dag[encoder_pipeline] = vi_extractor

        dag[dag_pipeline.DagOutput(mode + '_vimusic')] = encoder_pipeline

    return dag_pipeline.DAGPipeline(dag)