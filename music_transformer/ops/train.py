def run_training(build_graph_fn, train_dir, num_training_steps=None,
                 summary_frequency=10, save_checkpoint_secs=60,
                 checkpoints_to_keep=10, keep_checkpoint_every_n_hours=1,
                 master='', task=0, num_ps_tasks=0):
  """Runs the training loop.

  Args:
    build_graph_fn: A function that builds the graph ops.
    train_dir: The path to the directory where checkpoints and summary events
        will be written to.
    num_training_steps: The number of steps to train for before exiting.
    summary_frequency: The number of steps between each summary. A summary is
        when graph values from the last step are logged to the console and
        written to disk.
    save_checkpoint_secs: The frequency at which to save checkpoints, in
        seconds.
    checkpoints_to_keep: The number of most recent checkpoints to keep in
       `train_dir`. Keeps all if set to 0.
    keep_checkpoint_every_n_hours: Keep a checkpoint every N hours, even if it
        results in more checkpoints than checkpoints_to_keep.
    master: URL of the Tensorflow master.
    task: Task number for this worker.
    num_ps_tasks: Number of parameter server tasks.
  """
  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(num_ps_tasks)):
      build_graph_fn()

      global_step = tf.train.get_or_create_global_step()
      loss = tf.get_collection('loss')[0]
      perplexity = tf.get_collection('metrics/perplexity')[0]
      accuracy = tf.get_collection('metrics/accuracy')[0]
      train_op = tf.get_collection('train_op')[0]

      logging_dict = {
          'Global Step': global_step,
          'Loss': loss,
          'Perplexity': perplexity,
          'Accuracy': accuracy
      }
      hooks = [
          tf.train.NanTensorHook(loss),
          tf.train.LoggingTensorHook(
              logging_dict, every_n_iter=summary_frequency),
          tf.train.StepCounterHook(
              output_dir=train_dir, every_n_steps=summary_frequency)
      ]
      if num_training_steps:
        hooks.append(tf.train.StopAtStepHook(num_training_steps))

      scaffold = tf.train.Scaffold(
          saver=tf.train.Saver(
              max_to_keep=checkpoints_to_keep,
              keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))

      tf.logging.info('Starting training loop...')
      contrib_training.train(
          train_op=train_op,
          logdir=train_dir,
          scaffold=scaffold,
          hooks=hooks,
          save_checkpoint_secs=save_checkpoint_secs,
          save_summaries_steps=summary_frequency,
          master=master,
          is_chief=task == 0)
      tf.logging.info('Training complete.')