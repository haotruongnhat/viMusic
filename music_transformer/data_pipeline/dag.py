from magenta.music.protobuf import music_pb2
from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipelines_common

def get_pipeline(config, min_events, max_events, eval_ratio):
    """Returns the Pipeline instance for the music transformer

    Args:
        config: A Music Transformer config.
        min_events: Minimum number of events for an extracted sequence.
        max_events: Maximum number of events for an extracted sequence.
        eval_ratio: Fraction of input to set aside for evaluation set.

    Returns:
        A pipeline.Pipeline instance.
    """

    partitioner = pipelines_common.RandomPartition(
    music_pb2.NoteSequence,
    ['eval_performances', 'training_performances'],
    [eval_ratio])
    dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

    for mode in ['eval', 'training']:
        #applying pedal control change for the piano
        sustain_pipeline = note_sequence_pipelines.SustainPipeline(
        name='SustainPipeline_' + mode)

        #split note sequence at a regular interval
        splitter = note_sequence_pipelines.Splitter(
        hop_size_seconds=30.0, name='Splitter_' + mode)

        quantizer = note_sequence_pipelines.Quantizer(
        steps_per_second=config.steps_per_second, name='Quantizer_' + mode)

        perf_with_ly_extractor = PerformanceWithLyricsExtractor(
        min_events=min_events, max_events=max_events,
        num_velocity_bins=config.num_velocity_bins,
        name='PerformanceWithLyricsExtractor_' + mode)

        encoder_pipeline = ViMusicEncoderPipeline(config, name='ViMusicEncoderPipeline_' + mode)

    dag[sustain_pipeline] = partitioner[mode + '_performances']
    dag[splitter] = sustain_pipeline
    dag[quantizer] = splitter
    dag[perf_with_ly_extractor] = quantizer
    dag[encoder_pipeline] = perf_with_ly_extractor
    dag[dag_pipeline.DagOutput(mode + '_performances')] = encoder_pipeline

    return dag_pipeline.DAGPipeline(dag)

