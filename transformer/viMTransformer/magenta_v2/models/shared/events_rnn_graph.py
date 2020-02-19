# Copyright 2019 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides function to build an event sequence RNN model's graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import magenta
import numpy as np
import six
import tensorflow as tf
from tensorflow.contrib import cudnn_rnn as contrib_cudnn_rnn
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib import slim as contrib_slim
from tensorflow.python.util import nest as tf_nest


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=contrib_rnn.BasicLSTMCell,
                  residual_connections=False):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    residual_connections: Whether or not to use residual connections (via
        tf.contrib.rnn.ResidualWrapper).

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for i in range(len(rnn_layer_sizes)):
    cell = base_cell(rnn_layer_sizes[i])
    if attn_length and not cells:
      # Add attention wrapper to first layer.
      cell = contrib_rnn.AttentionCellWrapper(
          cell, attn_length, state_is_tuple=True)
    if residual_connections:
      cell = contrib_rnn.ResidualWrapper(cell)
      if i == 0 or rnn_layer_sizes[i] != rnn_layer_sizes[i - 1]:
        cell = contrib_rnn.InputProjectionWrapper(cell, rnn_layer_sizes[i])
    cell = contrib_rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = contrib_rnn.MultiRNNCell(cells)

  return cell


def state_tuples_to_cudnn_lstm_state(lstm_state_tuples):
  """Convert LSTMStateTuples to CudnnLSTM format."""
  h = tf.stack([s.h for s in lstm_state_tuples])
  c = tf.stack([s.c for s in lstm_state_tuples])
  return (h, c)


def cudnn_lstm_state_to_state_tuples(cudnn_lstm_state):
  """Convert CudnnLSTM format to LSTMStateTuples."""
  h, c = cudnn_lstm_state
  return tuple(
      contrib_rnn.LSTMStateTuple(h=h_i, c=c_i)
      for h_i, c_i in zip(tf.unstack(h), tf.unstack(c)))


def make_cudnn(inputs, rnn_layer_sizes, batch_size, mode,
               dropout_keep_prob=1.0, residual_connections=False):
  """Builds a sequence of cuDNN LSTM layers from the given hyperparameters.

  Args:
    inputs: A tensor of RNN inputs.
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    batch_size: The number of examples per batch.
    mode: 'train', 'eval', or 'generate'. For 'generate',
        CudnnCompatibleLSTMCell will be used.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    residual_connections: Whether or not to use residual connections.

  Returns:
    outputs: A tensor of RNN outputs, with shape
        `[batch_size, inputs.shape[1], rnn_layer_sizes[-1]]`.
    initial_state: The initial RNN states, a tuple with length
        `len(rnn_layer_sizes)` of LSTMStateTuples.
    final_state: The final RNN states, a tuple with length
        `len(rnn_layer_sizes)` of LSTMStateTuples.
  """
  cudnn_inputs = tf.transpose(a=inputs, perm=[1, 0, 2])

  if len(set(rnn_layer_sizes)) == 1 and not residual_connections:
    initial_state = tuple(
        contrib_rnn.LSTMStateTuple(
            h=tf.zeros([batch_size, num_units], dtype=tf.float32),
            c=tf.zeros([batch_size, num_units], dtype=tf.float32))
        for num_units in rnn_layer_sizes)

    if mode != 'generate':
      # We can make a single call to CudnnLSTM since all layers are the same
      # size and we aren't using residual connections.
      cudnn_initial_state = state_tuples_to_cudnn_lstm_state(initial_state)
      cell = contrib_cudnn_rnn.CudnnLSTM(
          num_layers=len(rnn_layer_sizes),
          num_units=rnn_layer_sizes[0],
          direction='unidirectional',
          dropout=1.0 - dropout_keep_prob)
      cudnn_outputs, cudnn_final_state = cell(
          cudnn_inputs, initial_state=cudnn_initial_state,
          training=mode == 'train')
      final_state = cudnn_lstm_state_to_state_tuples(cudnn_final_state)

    else:
      # At generation time we use CudnnCompatibleLSTMCell.
      cell = contrib_rnn.MultiRNNCell([
          contrib_cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
          for num_units in rnn_layer_sizes
      ])
      cudnn_outputs, final_state = tf.compat.v1.nn.dynamic_rnn(
          cell, cudnn_inputs, initial_state=initial_state, time_major=True,
          scope='cudnn_lstm/rnn')

  else:
    # We need to make multiple calls to CudnnLSTM, keeping the initial and final
    # states at each layer.
    initial_state = []
    final_state = []

    for i in range(len(rnn_layer_sizes)):
      # If we're using residual connections and this layer is not the same size
      # as the previous layer, we need to project into the new size so the
      # (projected) input can be added to the output.
      if residual_connections:
        if i == 0 or rnn_layer_sizes[i] != rnn_layer_sizes[i - 1]:
          cudnn_inputs = contrib_layers.linear(cudnn_inputs, rnn_layer_sizes[i])

      layer_initial_state = (contrib_rnn.LSTMStateTuple(
          h=tf.zeros([batch_size, rnn_layer_sizes[i]], dtype=tf.float32),
          c=tf.zeros([batch_size, rnn_layer_sizes[i]], dtype=tf.float32)),)

      if mode != 'generate':
        cudnn_initial_state = state_tuples_to_cudnn_lstm_state(
            layer_initial_state)
        cell = contrib_cudnn_rnn.CudnnLSTM(
            num_layers=1,
            num_units=rnn_layer_sizes[i],
            direction='unidirectional',
            dropout=1.0 - dropout_keep_prob)
        cudnn_outputs, cudnn_final_state = cell(
            cudnn_inputs, initial_state=cudnn_initial_state,
            training=mode == 'train')
        layer_final_state = cudnn_lstm_state_to_state_tuples(cudnn_final_state)

      else:
        # At generation time we use CudnnCompatibleLSTMCell.
        cell = contrib_rnn.MultiRNNCell(
            [contrib_cudnn_rnn.CudnnCompatibleLSTMCell(rnn_layer_sizes[i])])
        cudnn_outputs, layer_final_state = tf.compat.v1.nn.dynamic_rnn(
            cell, cudnn_inputs, initial_state=layer_initial_state,
            time_major=True,
            scope='cudnn_lstm/rnn' if i == 0 else 'cudnn_lstm_%d/rnn' % i)

      if residual_connections:
        cudnn_outputs += cudnn_inputs

      cudnn_inputs = cudnn_outputs

      initial_state += layer_initial_state
      final_state += layer_final_state

  outputs = tf.transpose(a=cudnn_outputs, perm=[1, 0, 2])

  return outputs, tuple(initial_state), tuple(final_state)


def get_build_graph_fn(mode, config, sequence_example_file_paths=None):
  """Returns a function that builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
        to use.
    sequence_example_file_paths: A list of paths to TFRecord files containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation.

  Returns:
    A function that builds the TF ops when called.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError("The mode parameter must be 'train', 'eval', "
                     "or 'generate'. The mode parameter was: %s" % mode)

  hparams = config.hparams
  encoder_decoder = config.encoder_decoder

  if hparams.use_cudnn and hparams.attn_length:
    raise ValueError('Using attention with cuDNN not currently supported.')

  tf.compat.v1.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size
  num_classes = encoder_decoder.num_classes
  no_event_label = encoder_decoder.default_event_label

  def build():
    """Builds the Tensorflow graph."""
    inputs, labels, lengths = None, None, None

    if mode in ('train', 'eval'):
      if isinstance(no_event_label, numbers.Number):
        label_shape = []
      else:
        label_shape = [len(no_event_label)]
      inputs, labels, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams.batch_size, input_size,
          label_shape=label_shape, shuffle=mode == 'train')

    elif mode == 'generate':
      inputs = tf.compat.v1.placeholder(tf.float32, [hparams.batch_size, None,
                                           input_size])

    if isinstance(encoder_decoder,
                  magenta.music.OneHotIndexEventSequenceEncoderDecoder):
      expanded_inputs = tf.one_hot(
          tf.cast(tf.squeeze(inputs, axis=-1), tf.int64),
          encoder_decoder.input_depth)
    else:
      expanded_inputs = inputs

    dropout_keep_prob = 1.0 if mode == 'generate' else hparams.dropout_keep_prob

    if hparams.use_cudnn:
      outputs, initial_state, final_state = make_cudnn(
          expanded_inputs, hparams.rnn_layer_sizes, hparams.batch_size, mode,
          dropout_keep_prob=dropout_keep_prob,
          residual_connections=hparams.residual_connections)

    else:
      cell = make_rnn_cell(
          hparams.rnn_layer_sizes,
          dropout_keep_prob=dropout_keep_prob,
          attn_length=hparams.attn_length,
          residual_connections=hparams.residual_connections)

      initial_state = cell.zero_state(hparams.batch_size, tf.float32)

      outputs, final_state = tf.compat.v1.nn.dynamic_rnn(
          cell, inputs, sequence_length=lengths, initial_state=initial_state,
          swap_memory=True)

    outputs_flat = magenta.common.flatten_maybe_padded_sequences(
        outputs, lengths)
    if isinstance(num_classes, numbers.Number):
      num_logits = num_classes
    else:
      num_logits = sum(num_classes)
    logits_flat = contrib_layers.linear(outputs_flat, num_logits)

    if mode in ('train', 'eval'):
      labels_flat = magenta.common.flatten_maybe_padded_sequences(
          labels, lengths)

      if isinstance(num_classes, numbers.Number):
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_flat, logits=logits_flat)
        predictions_flat = tf.argmax(input=logits_flat, axis=1)
      else:
        logits_offsets = np.cumsum([0] + num_classes)
        softmax_cross_entropy = []
        predictions = []
        for i in range(len(num_classes)):
          softmax_cross_entropy.append(
              tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=labels_flat[:, i],
                  logits=logits_flat[
                      :, logits_offsets[i]:logits_offsets[i + 1]]))
          predictions.append(
              tf.argmax(input=logits_flat[
                  :, logits_offsets[i]:logits_offsets[i + 1]], axis=1))
        predictions_flat = tf.stack(predictions, 1)

      correct_predictions = tf.cast(
          tf.equal(labels_flat, predictions_flat), dtype=tf.float32)
      event_positions = tf.cast(tf.not_equal(labels_flat, no_event_label), dtype=tf.float32)
      no_event_positions = tf.cast(tf.equal(labels_flat, no_event_label), dtype=tf.float32)

      # Compute the total number of time steps across all sequences in the
      # batch. For some models this will be different from the number of RNN
      # steps.
      def batch_labels_to_num_steps(batch_labels, lengths):
        num_steps = 0
        for labels, length in zip(batch_labels, lengths):
          num_steps += encoder_decoder.labels_to_num_steps(labels[:length])
        return np.float32(num_steps)
      num_steps = tf.compat.v1.py_func(
          batch_labels_to_num_steps, [labels, lengths], tf.float32)

      if mode == 'train':
        loss = tf.reduce_mean(input_tensor=softmax_cross_entropy)
        perplexity = tf.exp(loss)
        accuracy = tf.reduce_mean(input_tensor=correct_predictions)
        event_accuracy = (
            tf.reduce_sum(input_tensor=correct_predictions * event_positions) /
            tf.reduce_sum(input_tensor=event_positions))
        no_event_accuracy = (
            tf.reduce_sum(input_tensor=correct_predictions * no_event_positions) /
            tf.reduce_sum(input_tensor=no_event_positions))

        loss_per_step = tf.reduce_sum(input_tensor=softmax_cross_entropy) / num_steps
        perplexity_per_step = tf.exp(loss_per_step)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=hparams.learning_rate)

        train_op = contrib_slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparams.clip_norm)
        tf.compat.v1.add_to_collection('train_op', train_op)

        vars_to_summarize = {
            'loss': loss,
            'metrics/perplexity': perplexity,
            'metrics/accuracy': accuracy,
            'metrics/event_accuracy': event_accuracy,
            'metrics/no_event_accuracy': no_event_accuracy,
            'metrics/loss_per_step': loss_per_step,
            'metrics/perplexity_per_step': perplexity_per_step,
        }
      elif mode == 'eval':
        vars_to_summarize, update_ops = contrib_metrics.aggregate_metric_map({
            'loss':
                tf.compat.v1.metrics.mean(softmax_cross_entropy),
            'metrics/accuracy':
                tf.compat.v1.metrics.accuracy(labels_flat, predictions_flat),
            'metrics/per_class_accuracy':
                tf.compat.v1.metrics.mean_per_class_accuracy(labels_flat,
                                                   predictions_flat,
                                                   num_classes),
            'metrics/event_accuracy':
                tf.compat.v1.metrics.recall(event_positions, correct_predictions),
            'metrics/no_event_accuracy':
                tf.compat.v1.metrics.recall(no_event_positions, correct_predictions),
            'metrics/loss_per_step':
                tf.compat.v1.metrics.mean(
                    tf.reduce_sum(input_tensor=softmax_cross_entropy) / num_steps,
                    weights=num_steps),
        })
        for updates_op in update_ops.values():
          tf.compat.v1.add_to_collection('eval_ops', updates_op)

        # Perplexity is just exp(loss) and doesn't need its own update op.
        vars_to_summarize['metrics/perplexity'] = tf.exp(
            vars_to_summarize['loss'])
        vars_to_summarize['metrics/perplexity_per_step'] = tf.exp(
            vars_to_summarize['metrics/loss_per_step'])

      for var_name, var_value in six.iteritems(vars_to_summarize):
        tf.compat.v1.summary.scalar(var_name, var_value)
        tf.compat.v1.add_to_collection(var_name, var_value)

    elif mode == 'generate':
      temperature = tf.compat.v1.placeholder(tf.float32, [])
      if isinstance(num_classes, numbers.Number):
        softmax_flat = tf.nn.softmax(
            tf.compat.v1.div(logits_flat, tf.fill([num_classes], temperature)))
        softmax = tf.reshape(
            softmax_flat, [hparams.batch_size, -1, num_classes])
      else:
        logits_offsets = np.cumsum([0] + num_classes)
        softmax = []
        for i in range(len(num_classes)):
          sm = tf.nn.softmax(
              tf.compat.v1.div(
                  logits_flat[:, logits_offsets[i]:logits_offsets[i + 1]],
                  tf.fill([num_classes[i]], temperature)))
          sm = tf.reshape(sm, [hparams.batch_size, -1, num_classes[i]])
          softmax.append(sm)

      tf.compat.v1.add_to_collection('inputs', inputs)
      tf.compat.v1.add_to_collection('temperature', temperature)
      tf.compat.v1.add_to_collection('softmax', softmax)
      # Flatten state tuples for metagraph compatibility.
      for state in tf_nest.flatten(initial_state):
        tf.compat.v1.add_to_collection('initial_state', state)
      for state in tf_nest.flatten(final_state):
        tf.compat.v1.add_to_collection('final_state', state)

  return build
