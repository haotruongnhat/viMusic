import magenta.music as mm
import magenta
from magenta.models.performance_rnn.performance_sequence_generator import *
from magenta.models.performance_rnn import performance_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from magenta.music.protobuf import generator_pb2
from magenta.music.midi_io import *
from magenta.models.performance_rnn import performance_model
import tensorflow as tf
import ast, time, os

def generate_melody(dataset_name,config,second=10,beam_size=1,branch_factor=1,primer_melody="[60,62,64,65,67,69,71,72]",
steps_per_iteration=1,temperature=1.0,num_outputs=1):
    checkpoint = os.path.join("./performance_rnn/run/" + dataset_name,config + "/train")
    output_dir = os.path.join("./performance_rnn/generated/",dataset_name + "/" + config)
    num_steps= second * 600
    """Saves bundle or runs generator based on flags."""
    tf.logging.set_verbosity('INFO')
    config_model = performance_model.default_configs[config]
    config_model.hparams.parse('')
    # Having too large of a batch size will slow generation down unnecessarily.
    config_model.hparams.batch_size = min(config_model.hparams.batch_size, beam_size * branch_factor)
    generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
      model=performance_model.PerformanceRnnModel(config_model),
      details=config_model.details,
      steps_per_second=config_model.steps_per_second,
      num_velocity_bins=config_model.num_velocity_bins,
      control_signals=config_model.control_signals,
      optional_conditioning=config_model.optional_conditioning,
      checkpoint=checkpoint,
      bundle=None,
      note_performance=config_model.note_performance)

    primer_sequence = magenta.music.Melody(ast.literal_eval(primer_melody)).to_sequence()

    seconds_per_step = 1.0 / generator.steps_per_second
    generate_end_time = num_steps * seconds_per_step

    # Specify start/stop time for generation based on starting generation at the
    # end of the priming sequence and continuing until the sequence is num_steps
    # long.
    generator_options = generator_pb2.GeneratorOptions()
    # Set the start time to begin when the last note ends.
    generate_section = generator_options.generate_sections.add(
        start_time=primer_sequence.total_time,
        end_time=generate_end_time)

    if generate_section.start_time >= generate_section.end_time:
        tf.logging.fatal(
            'Priming sequence is longer than the total number of steps '
            'requested: Priming sequence length: %s, Total length '
            'requested: %s',
            generate_section.start_time, generate_end_time)


    generator_options.args['temperature'].float_value = temperature
    generator_options.args['beam_size'].int_value = beam_size
    generator_options.args['branch_factor'].int_value = branch_factor
    generator_options.args['steps_per_iteration'].int_value = steps_per_iteration

    tf.logging.debug('primer_sequence: %s', primer_sequence)
    tf.logging.debug('generator_options: %s', generator_options)

    # Make the generate request num_outputs times and save the output as midi
    # files.
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    digits = len(str(num_outputs))
    for i in range(num_outputs):
        generated_sequence = generator.generate(primer_sequence, generator_options)

        midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
        midi_path = os.path.join(output_dir, midi_filename)
        magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

    tf.logging.info('Wrote %d MIDI files to %s',
                    num_outputs, output_dir)