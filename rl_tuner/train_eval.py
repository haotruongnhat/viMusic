import os, magenta
from magenta.models.melody_rnn import melody_rnn_model
from magenta.music.protobuf import generator_pb2

from magenta.models.rl_tuner import rl_tuner
from magenta.models.rl_tuner import rl_tuner_ops
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_train
import matplotlib
import matplotlib.pyplot as plt  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow.contrib import training as contrib_training

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
# Number of output note classes. This is a property of the dataset.
NUM_CLASSES = 38

#define some parameters
def train_rl_tuner(dataset_name):
      output_dir = './rl_tuner/run/' + dataset_name + '/rl_tuner/train'
      note_rnn_checkpoint_dir = './rl_tuner/run/' + dataset_name + '/note_rnn/train'
      note_rnn_type = 'basic_rnn' #key idea of note rnn
      training_steps = 10000
      exploration_steps = 5000
      exploration_mode = 'boltzmann'
      output_every_nth = 50 #output checkpoint every nth step
      num_notes_in_melody = 32
      reward_scalar = 0.1
      algorithm = 'g' #q, psi, g
      layer_size = 512
      batch_size = 1

      #hparams for basic rnn
      #layer sizes: 38, 4 layer due to the melody rnn
      hparams = contrib_training.HParams(
      batch_size=batch_size, rnn_layer_sizes=layer_size, one_hot_length=NUM_CLASSES,
      use_cudnn=True)

      #params for dqn
      dqn_hparams = contrib_training.HParams(
            batch_size=batch_size,
            random_action_probability=0.1,
            store_every_nth=1,
            train_every_nth=5,
            minibatch_size=1, #hard code: need to recheck
            discount_rate=0.5,
            max_experience=100000,
            target_network_update_rate=0.01,
            use_cdnn=True)


      output_dir = os.path.join(output_dir,algorithm)
      output_ckpt = algorithm + '.ckpt'

      rlt = rl_tuner.RLTuner(output_dir,
      midi_primer=None,
      dqn_hparams=dqn_hparams,
      reward_scaler=reward_scalar,
      save_name=output_ckpt,
      output_every_nth=output_every_nth,
      note_rnn_checkpoint_dir=note_rnn_checkpoint_dir,
      note_rnn_checkpoint_file=None,
      note_rnn_type=note_rnn_type,
      note_rnn_hparams=hparams,
      num_notes_in_melody=num_notes_in_melody,
      exploration_mode=exploration_mode,
      algorithm=algorithm)

      rlt.train(num_steps=training_steps,
            exploration_period=exploration_steps)

      for i in range(0,10):
            rlt.generate_music_sequence(visualize_probs=False,length=4000,title='rl_tuner_song')
      rlt.save_model(output_dir, '.')



def train_note_rnn(dataset_name):
      run_dir = './rl_tuner/run/' + dataset_name + '/note_rnn'
      sequence_training_file = './rl_tuner/tmp_dataset/' + dataset_name + '/note_rnn/training_melodies.tfrecord'
      num_training_steps = 500
      summary_frequency = 10
      num_checkpoints = 50
      tf.logging.set_verbosity('INFO')

      sequence_training_file = tf.gfile.Glob(
            os.path.expanduser(sequence_training_file)
      )

      run_dir = os.path.expanduser(run_dir)

      #customize hparams
      hparams = contrib_training.HParams(
      batch_size=128, 
      rnn_layer_sizes=512, 
      one_hot_length=NUM_CLASSES)
      #Use cudnn for basic RNN

      #create basic RNN config model
      config_model = melody_rnn_model.MelodyRnnConfig(
        generator_pb2.GeneratorDetails(
            id='basic_rnn',
            description='Melody RNN with one-hot encoding.'),
        magenta.music.OneHotEventSequenceEncoderDecoder(
            magenta.music.MelodyOneHotEncoding(
                min_note=DEFAULT_MIN_NOTE,
                max_note=DEFAULT_MAX_NOTE)),
            hparams)
      
      #create graph for basic rnn
      build_graph_fn = events_rnn_graph.get_build_graph_fn(
      'train', config_model, sequence_training_file)

      train_dir = os.path.join(run_dir, 'train')
      if not os.path.exists(train_dir):
            tf.gfile.MakeDirs(train_dir)

      #begin traning
      events_rnn_train.run_training(build_graph_fn, train_dir,
            num_training_steps,
            summary_frequency,
            checkpoints_to_keep=num_checkpoints)