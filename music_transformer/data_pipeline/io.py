
import os

from music_transformer.music import musicxml_reader
from magenta.music import note_sequence_io
from magenta.pipelines import pipeline
import tensorflow as tf

from .dag import get_pipeline

from music_transformer.utils import default_config
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
    #midi conversion with lyrics is not supported yet!
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
                pass
                #sequence = convert_midi(root_dir, sub_dir, full_file_path)
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

def convert_directory(root_dir, output_file, recursive=False):
    """Converts files to NoteSequences and writes to `output_file`.

    Input files found in `root_dir` are converted to NoteSequence protos with the
    basename of `root_dir` as the collection_name, and the relative path to the
    file from `root_dir` as the filename. If `recursive` is true, recursively
    converts any subdirectories of the specified directory.

    Args:
        root_dir: A string specifying a root directory.
        output_file: Path to TFRecord file to write results to.
        recursive: A boolean specifying whether or not recursively convert files
            contained in subdirectories of the specified directory.
    """
    with note_sequence_io.NoteSequenceRecordWriter(output_file) as writer:
        convert_files(root_dir, '', writer, recursive)


###############Function to transform data using pipeline
def gen_tf_dataset_from_tfrecord(input_tfrecord,output_tfrecord,config=default_config,eval_ratio=0.1):
    pipeline_instance = get_pipeline(config,eval_ratio)
    pipeline.run_pipeline_serial(
        pipeline_instance,
        pipeline.tf_record_iterator(input_tfrecord,pipeline_instance.input_type),
        output_tfrecord
    )

