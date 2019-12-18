from magenta.models.shared import events_rnn_graph
from magenta.models.shared import events_rnn_train
from magenta.models.performance_rnn import performance_model
import tensorflow as tf
import os, magenta, glob

def eval_performance_rnn(dataset_name,config,hparams='',num_eval_examples = 0):
    run_dir = "./performance_rnn/run/" + dataset_name + "/" + config
    sequence_training_file = './performance_rnn/tmp_dataset/' + dataset_name + '/training_performances.tfrecord'
    sequence_training_file_paths = tf.gfile.Glob(
    os.path.expanduser(sequence_training_file))

    sequence_eval_file = './performance_rnn/tmp_dataset/' + dataset_name + '/eval_performances.tfrecord'
    sequence_eval_file_paths = tf.gfile.Glob(
        os.path.expanduser(sequence_eval_file))

    run_dir = os.path.expanduser(run_dir)
    
    config_model = performance_model.default_configs[config]
    config_model.hparams.parse(hparams) #not specified yet
    
    build_graph_fn = events_rnn_graph.get_build_graph_fn(
      'train', config_model, sequence_training_file_paths)
      
    train_dir = os.path.join(run_dir, 'train')
    tf.gfile.MakeDirs(train_dir)
    tf.logging.info('Train dir: %s', train_dir)
    
    eval_dir = os.path.join(run_dir, 'eval')
    train_dir = os.path.join(run_dir, 'train')
    tf.gfile.MakeDirs(eval_dir)
    tf.logging.info('Eval dir: %s', eval_dir)

    num_batches = (
        (num_eval_examples or
        magenta.common.count_records(sequence_eval_file_paths)) //
        config_model.hparams.batch_size)
    events_rnn_train.run_eval(build_graph_fn, train_dir, eval_dir, num_batches)

def train_performance_rnn(dataset_name,config,continue_train=True,hparams='', training_steps = 12000, summary_frequency = 10, num_checkpoints=100):
    run_dir = "./performance_rnn/run/" + dataset_name + "/" + config
    sequence_training_file = './performance_rnn/tmp_dataset/' + dataset_name + '/training_performances.tfrecord'
    
    sequence_training_file_paths = tf.gfile.Glob(
    os.path.expanduser(sequence_training_file))
    run_dir = os.path.expanduser(run_dir)
    
    config_model = performance_model.default_configs[config]
    config_model.hparams.parse(hparams) #not specified yet
    
    build_graph_fn = events_rnn_graph.get_build_graph_fn(
      'train', config_model, sequence_training_file_paths)
      
    train_dir = os.path.join(run_dir, 'train')
    tf.gfile.MakeDirs(train_dir)
    tf.logging.info('Train dir: %s', train_dir)

    if not continue_train:
        files = glob.glob(os.path.join(train_dir,'*'))
        for f in files:
            os.remove(f)
    
    events_rnn_train.run_training(build_graph_fn, train_dir,
                                  training_steps,
                                  summary_frequency,
                                  checkpoints_to_keep=num_checkpoints)
