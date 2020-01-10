import ast
import os
import time

import magenta
from magenta.models.improv_rnn import improv_rnn_config_flags
from magenta.models.improv_rnn import improv_rnn_model
from magenta.models.improv_rnn import improv_rnn_sequence_generator
from magenta.music.protobuf import generator_pb2
from magenta.music.protobuf import music_pb2
from magenta.models.shared import sequence_generator
import tensorflow as tf
import magenta.music as mm
from magenta.pipelines import chord_pipelines
from magenta.pipelines import melody_pipelines
from vimusic_rnn.pipeline import ViMusicExtractor

from magenta.music import constants

from vimusic_rnn.utils import default_vimusic_configuration

from vimusic_rnn.utils import ViMusicRnnModel

from vimusic_rnn.lib import ViMusic

import functools, math

CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL

# Velocity at which to play chord notes when rendering chords.
CHORD_VELOCITY = 50

# This model can leave hanging notes. To avoid cacophony we turn off any note
# after 5 seconds.
MAX_NOTE_DURATION_SECONDS = 5.0
DEFAULT_NOTE_DENSITY = 15.0

def get_checkpoint(run_dir):
    if run_dir:
        train_dir = os.path.join(os.path.expanduser(run_dir), 'train')
        return train_dir
    else:
        return None

def _step_to_value(step, num_steps, values):
    """Map step in performance to desired control signal value."""
    num_segments = len(values)
    index = min(step * num_segments // num_steps, num_segments - 1)
    return values[index]

class ViMusicRnnSequenceGenerator(sequence_generator.BaseSequenceGenerator):
    def __init__(self, model, details, 
    steps_per_second=mm.DEFAULT_STEPS_PER_SECOND,
    num_velocity_bins=0,
    control_signals=None,
    optional_conditioning=False,
    max_note_duration=MAX_NOTE_DURATION_SECONDS,
    fill_generate_section=True,
    checkpoint=None,
    bundle=None):

        super(ViMusicRnnSequenceGenerator, self).__init__( \
        model, details, checkpoint, bundle)
        self.steps_per_second = steps_per_second
        self.num_velocity_bins = num_velocity_bins
        self.control_signals = control_signals
        self.optional_conditioning = optional_conditioning
        self.max_note_duration = max_note_duration
        self.fill_generate_section = fill_generate_section

    def _generate(self, input_sequence, generator_options):
        if len(generator_options.input_sections) > 1:
            raise sequence_generator.SequenceGeneratorError(
            'This model supports at most one input_sections message, but got %s' %
            len(generator_options.input_sections))
        if len(generator_options.generate_sections) != 1:
            raise sequence_generator.SequenceGeneratorError(
            'This model supports only 1 generate_sections message, but got %s' %
            len(generator_options.generate_sections))

        #qpm
        generate_section = generator_options.generate_sections[0]
        if generator_options.input_sections:
            input_section = generator_options.input_sections[0]
            primer_sequence = mm.trim_note_sequence(
                input_sequence, input_section.start_time, input_section.end_time)
            input_start_step = mm.quantize_to_step(
                input_section.start_time, self.steps_per_second, quantize_cutoff=0.0
            )
        else:
            primer_sequence = input_sequence
            input_start_step = 0

        if primer_sequence.notes:
            last_end_time = max(n.end_time for n in primer_sequence.notes)
        else:
            last_end_time = 0

        #from pdb import set_trace ; set_trace()
        if generate_section.start_time >= generate_section.end_time:
            raise sequence_generator.SequenceGeneratorError(
            'Got GenerateSection request for section that is before or equal to '
            'the end of the input section. This model can only extend melodies. '
            'Requested start time: %s, Final note end time: %s' %
            (generate_section.start_time, generate_section.start_time))

        # Quantize the priming and backing sequences.
        quantized_primer_sequence = mm.quantize_note_sequence_absolute(
            primer_sequence, self.steps_per_second)

        # Setting gap_bars to infinite ensures that the entire input will be used.

        extracted_vimusics, _ = ViMusicExtractor.extract_vimusic(
        quantized_primer_sequence,
        split_instruments=True, 
        start_step=input_start_step,
        num_velocity_bins=self.num_velocity_bins)
        assert len(extracted_vimusics) <= 1

        generate_start_step = mm.quantize_to_step(
        generate_section.start_time, self.steps_per_second, quantize_cutoff=0.0)
        # Note that when quantizing end_step, we set quantize_cutoff to 1.0 so it
        # always rounds down. This avoids generating a sequence that ends at 5.0
        # seconds when the requested end time is 4.99.
        generate_end_step = mm.quantize_to_step(
        generate_section.end_time, self.steps_per_second, quantize_cutoff=1.0)

        #get the first vimusic
        if extracted_vimusics and extracted_vimusics[0]:
            vimusic = extracted_vimusics[0]
        else:
            # If no track could be extracted, create an empty track that starts at the
            # requested generate_start_step.
            vimusic = ViMusic(
            steps_per_second=(
                quantized_primer_sequence.quantization_info.steps_per_second),
            start_step=generate_start_step,
            num_velocity_bins=self.num_velocity_bins)

        # Ensure that the track extends up to the step we want to start generating.
        vimusic.set_length(generate_start_step - vimusic.start_step)

        # Extract generation arguments from generator options.
        arg_types = {
        'temperature': lambda arg: arg.float_value,
        'beam_size': lambda arg: arg.int_value,
        'branch_factor': lambda arg: arg.int_value,
        'steps_per_iteration': lambda arg: arg.int_value
        }

        #??
        #defining value types for the control signals?
        if self.control_signals:
            for control in self.control_signals:
                arg_types[control.name] = lambda arg: ast.literal_eval(arg.string_value)

        #note sequence will contain information in the generator_options.args
        args = dict((name, value_fn(generator_options.args[name]))
                    for name, value_fn in arg_types.items()
                    if name in generator_options.args)

        # Make sure control signals are present and convert to lists if necessary.
        if self.control_signals:
            for control in self.control_signals:
                if control.name not in args:
                    tf.logging.warning(
                    'Control value not specified, using default: %s = %s',
                    control.name, control.default_value)
                    args[control.name] = [control.default_value]
                elif control.validate(args[control.name]):
                    args[control.name] = [args[control.name]]
                else:
                    if not isinstance(args[control.name], list) or not all(
                    control.validate(value) for value in args[control.name]):
                        tf.logging.fatal(
                        'Invalid control value: %s = %s',
                        control.name, args[control.name])

        # Make sure disable conditioning flag is present when conditioning is
        # optional and convert to list if necessary.
        if self.optional_conditioning:
            if 'disable_conditioning' not in args:
                args['disable_conditioning'] = [False]
            elif isinstance(args['disable_conditioning'], bool):
                args['disable_conditioning'] = [args['disable_conditioning']]
            else:
                if not isinstance(args['disable_conditioning'], list) or not all(
                isinstance(value, bool) for value in args['disable_conditioning']):
                    tf.logging.fatal(
                    'Invalid disable_conditioning value: %s',
                    args['disable_conditioning'])


        total_steps = vimusic.num_steps + (
        generate_end_step - generate_start_step)

        if 'notes_per_second' in args:
            mean_note_density = (
            sum(args['notes_per_second']) / len(args['notes_per_second']))
        else:
            mean_note_density = DEFAULT_NOTE_DENSITY

        # Set up functions that map generation step to control signal values and
        # disable conditioning flag.
        if self.control_signals:
            control_signal_fns = []
            for control in self.control_signals:
                control_signal_fns.append(functools.partial(
                _step_to_value,
                num_steps=total_steps,
                values=args[control.name]))
                del args[control.name]
            args['control_signal_fns'] = control_signal_fns
        if self.optional_conditioning:
            args['disable_conditioning_fn'] = functools.partial(
            _step_to_value,
            num_steps=total_steps,
            values=args['disable_conditioning'])
            del args['disable_conditioning']
            
        if not vimusic:
            vimusic.set_length(min(vimusic.max_shift_steps, total_steps))

        while vimusic.num_steps < total_steps:
            note_density = max(1.0,mean_note_density)
            steps_to_gen = total_steps - vimusic.num_steps
            rnn_steps_to_gen = int(math.ceil(
            4.0 * note_density * steps_to_gen / self.steps_per_second))
            tf.logging.info(
            'Need to generate %d more steps for this sequence, will try asking '
            'for %d RNN steps' % (steps_to_gen, rnn_steps_to_gen))
            vimusic = self._model.generate_vimusic(
            len(vimusic.get_notes()) + rnn_steps_to_gen, vimusic, **args)

            if not self.fill_generate_section:
                break

        from pdb import set_trace ; set_trace()
        vimusic.set_length(total_steps)

        generated_sequence = vimusic.to_sequence(
            max_note_duration=self.max_note_duration
        )

        assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
        return generated_sequence


def generate_melody(dataset_name,
config=default_vimusic_configuration, 
duration_in_seconds=30,
beam_size=1,
branch_factor=1,
temperature=1.0,
steps_per_iteration = 1,
num_outputs=1):
    #specify and create directories
    run_dir = "./vimusic_rnn/run/" + dataset_name
    output_dir = "./vimusic_rnn/generated/" + dataset_name
    output_dir = os.path.expanduser(output_dir)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    #reconfigurating batch_size (due to size)
    config.hparams.batch_size = min(
    config.hparams.batch_size, beam_size * branch_factor)

    #Create sequence generator
    generator = ViMusicRnnSequenceGenerator(
        model=ViMusicRnnModel(config),
        details=config.details,
        steps_per_second=config.steps_per_second,
        num_velocity_bins=config.num_velocity_bins,
        control_signals=config.control_signals,
        optional_conditioning=config.optional_conditioning,
        checkpoint=get_checkpoint(run_dir)
    )

    #define primer sequence
    primer_pitches = [60, 64, 67, 120, 70]
    primer_velocities = [100, 90, 40, 120, 60]
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    for p, v in zip(primer_pitches, primer_velocities):
        note = primer_sequence.notes.add()
        note.start_time = 0
        note.end_time = 60.0 / magenta.music.DEFAULT_QUARTERS_PER_MINUTE
        note.pitch = p
        note.velocity = v
        primer_sequence.total_time = note.end_time 


    #create generator_options
    generator_options = generator_pb2.GeneratorOptions()
    generate_section = generator_options.generate_sections.add(
        start_time = primer_sequence.total_time, #begin from where primer sequence left
        end_time = duration_in_seconds - primer_sequence.total_time
    )

    if generate_section.start_time >= generate_section.end_time:
        tf.logging.fatal('Oops some bugs here :v!')
        returns

    #We will use all of the control signal in this step
    #If you want to remove any control signal, modify it on the [generator]
    generator_options.args['temperature'].float_value = temperature
    generator_options.args['beam_size'].int_value = beam_size
    generator_options.args['branch_factor'].int_value = branch_factor
    generator_options.args[
      'steps_per_iteration'].int_value = steps_per_iteration
    tf.logging.debug('primer_sequence: %s', primer_sequence)
    tf.logging.debug('generator_options: %s', generator_options)

    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    digits = len(str(num_outputs))
    for i in range(num_outputs):
        generated_sequence = generator.generate(primer_sequence, generator_options)

        midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
        midi_path = os.path.join(output_dir, midi_filename)
        magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

    tf.logging.info('Wrote %d MIDI files to %s',
                  num_outputs, output_dir)


