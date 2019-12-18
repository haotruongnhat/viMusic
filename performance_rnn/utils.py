import magenta, os
from magenta.music import note_sequence_io
from magenta.music import midi_io
from magenta.models.performance_rnn import performance_model
import tensorflow as tf
from magenta.pipelines import performance_pipeline
from magenta.pipelines import pipeline

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
    try:
        sequence = midi_io.midi_to_sequence_proto(tf.gfile.GFile(full_file_path, 'rb').read())
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

def convert_files(root_dir, sub_dir, writer, recursive=False):
    """Converts files.

    Args:
    root_dir: A string specifying a root directory.
    sub_dir: A string specifying a path to a directory under `root_dir` in which
        to convert contents.
    writer: A TFRecord writer
    recursive: A boolean specifying whether or not recursively convert files
        contained in subdirectories of the specified directory.

    Returns:
    A map from the resulting Futures to the file paths being converted.
    """
    dir_to_convert = os.path.join(root_dir, sub_dir)
    tf.logging.info("Converting files in '%s'.", dir_to_convert)
    files_in_dir = tf.gfile.ListDirectory(os.path.join(dir_to_convert))
    recurse_sub_dirs = []
    written_count = 0
    for file_in_dir in files_in_dir:
        tf.logging.log_every_n(tf.logging.INFO, '%d files converted.',
                           1000, written_count)
        full_file_path = os.path.join(dir_to_convert, file_in_dir)
        if (full_file_path.lower().endswith('.mid') or full_file_path.lower().endswith('.midi')):
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

#Read train data
def create_dataset(dataset_name,eval_ratio=0.1,config="performance_with_dynamics"):
    input_dir = "./datasets/" + dataset_name
    output_file = "./datasets/" + dataset_name + ".tfrecord"

    #first stage tfrecord creation
    with note_sequence_io.NoteSequenceRecordWriter(output_file) as writer:
        convert_files(input_dir, '', writer, True)

def gen_midi_dataset_by_model(dataset_name,eval_ratio=0.1,config="performance_with_dynamics"):
    input_tfrecord = "./datasets/" + dataset_name + ".tfrecord"
    output_tfrecord = './performance_rnn/tmp_dataset/' + dataset_name
    #only for performance and performance_with_dynamics

    pipeline_instance = performance_pipeline.get_pipeline(
      min_events=32,
      max_events=512,
      eval_ratio=eval_ratio,
      config=performance_model.default_configs[config])

    pipeline.run_pipeline_serial(
        pipeline_instance,
        pipeline.tf_record_iterator(input_tfrecord, pipeline_instance.input_type),
        output_tfrecord)