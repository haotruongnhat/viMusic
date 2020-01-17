import tensorflow as tf
import os, glob
from vimusic_rnn.utils import default_vimusic_configuration
#from vimusic_rnn.utils import get_graph_builder
from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_train

def train_vimusic_rnn(dataset_name,config=default_vimusic_configuration,hparams='', training_steps = 120000, summary_frequency=50, num_checkpoints=1):
    run_dir = "./vimusic_rnn/run/" + dataset_name
    sequence_training_file = './vimusic_rnn/tmp_dataset/' + dataset_name + '/train_vimusic.tfrecord'
    
    sequence_training_file_paths = tf.gfile.Glob(
    os.path.expanduser(sequence_training_file))
    run_dir = os.path.expanduser(run_dir)
    
    config.hparams.parse(hparams) #not specified yet
    
    build_graph_fn = events_rnn_graph.get_build_graph_fn(
      'train', config, sequence_training_file_paths)
      
    train_dir = os.path.join(run_dir, 'train')
    tf.gfile.MakeDirs(train_dir)
    tf.logging.info('Train dir: %s', train_dir)
    
    events_rnn_train.run_training(build_graph_fn, train_dir,
                                  training_steps,
                                  summary_frequency,
                                  checkpoints_to_keep=num_checkpoints)