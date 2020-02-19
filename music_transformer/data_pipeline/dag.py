from music_transformer.music import vimusic_pb2
from magenta.pipelines import dag_pipeline
from magenta.pipelines import pipelines_common
from magenta.pipelines import note_sequence_pipelines

from .pipelines import ViMusicWithLyricsExtractor
from .pipelines import ViMusicEncoderPipeline
from .pipelines import ViQuantizerPipeline
from .pipelines import UnifyNotesPipeline
from .pipelines import ViSustainPipeline
from .pipelines import ViSplitterPipeline

from music_transformer.utils.constants import *

def get_pipeline(config, eval_ratio):
    """Returns the Pipeline instance for the music transformer"""

    partitioner = pipelines_common.RandomPartition(
    vimusic_pb2.NoteSequence,
    ['eval_performances', 'training_performances'],
    [eval_ratio])
    dag = {partitioner: dag_pipeline.DagInput(vimusic_pb2.NoteSequence)}

    for mode in ['eval', 'training']:
        #applying pedal control change for the piano
        sustain_pipeline = ViSustainPipeline(
        name='SustainPipeline_' + mode)

        #split meldoy based on time signatures
        time_change_splitter = note_sequence_pipelines.TimeChangeSplitter(
        name='TimeChangeSplitter_' + mode)

        #split note sequence at a regular interval
        splitter = ViSplitterPipeline(
        hop_size_seconds=30.0, name='Splitter_' + mode)

        #applying chord symbol to notes
        unify_note_pipeline = UnifyNotesPipeline(DEFAULT_CHORD_BASE_PITCH
        ,name='UnifyNotesPipeline_' + mode
        )

        quantizer = ViQuantizerPipeline(
        steps_per_second=config.steps_per_second, name='Quantizer_' + mode)

        vi_with_ly_extractor = ViMusicWithLyricsExtractor(
        min_events=config.min_events, max_events=config.max_events,
        num_velocity_bins=config.num_velocity_bins,
        name='PerformanceWithLyricsExtractor_' + mode)

        encoder_pipeline = ViMusicEncoderPipeline(config,
        name='ViMusicEncoderPipeline_' + mode)

    dag[sustain_pipeline] = partitioner[mode + '_performances']
    dag[splitter] = sustain_pipeline
    dag[unify_note_pipeline] = splitter
    dag[quantizer] = unify_note_pipeline
    dag[vi_with_ly_extractor] = quantizer
    dag[encoder_pipeline] = vi_with_ly_extractor
    dag[dag_pipeline.DagOutput(mode + '_performances')] = encoder_pipeline

    return dag_pipeline.DAGPipeline(dag)

