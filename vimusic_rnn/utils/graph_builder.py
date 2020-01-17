import magenta
from .cudnn_creation import make_cudnn,make_cudnn_for_generation
from tensorflow.contrib import layers as contrib_layers
from .configuration import default_vimusic_configuration
import tensorflow as tf
import numbers
def get_graph_builder(mode, config=default_vimusic_configuration, sequence_example_file_paths=None):
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

    #initial configuraation assignment
    hparams = config.hparams
    encoder_decoder = config.encoder_decoder

    if hparams.use_cudnn and hparams.attn_length:
        raise ValueError('Using attention with cuDNN not currently supported.')

    tf.logging.info('hparams = %s', hparams.values())

    input_size = encoder_decoder.input_size
    num_classes = encoder_decoder.num_classes
    no_event_label = encoder_decoder.default_event_label

    #return function for building graph
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
            inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,input_size])

        if isinstance(encoder_decoder,
                    magenta.music.OneHotIndexEventSequenceEncoderDecoder):
            expanded_inputs = tf.one_hot(
                tf.cast(tf.squeeze(inputs, axis=-1), tf.int64),
                encoder_decoder.input_depth)
        else:
            expanded_inputs = inputs

        dropout_keep_prob = 1.0 if mode == 'generate' else hparams.dropout_keep_prob

        if mode in ('train','eval'):
            outputs, initial_state, final_state = make_cudnn(
            expanded_inputs, hparams.rnn_layer_sizes, hparams.batch_size,
            residual_connections=hparams.residual_connections)
        elif mode == 'generate':
            outputs, initial_state, final_state = make_cudnn_for_generation(
            expanded_inputs, hparams.rnn_layer_sizes, hparams.batch_size, mode,
            dropout_keep_prob=dropout_keep_prob,
            residual_connections=hparams.residual_connections)


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
                predictions_flat = tf.argmax(logits_flat, axis=1)
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
                        tf.argmax(logits_flat[
                            :, logits_offsets[i]:logits_offsets[i + 1]], axis=1))
                predictions_flat = tf.stack(predictions, 1)

        correct_predictions = tf.to_float(
            tf.equal(labels_flat, predictions_flat))
        event_positions = tf.to_float(tf.not_equal(labels_flat, no_event_label))
        no_event_positions = tf.to_float(tf.equal(labels_flat, no_event_label))

        # Compute the total number of time steps across all sequences in the
        # batch. For some models this will be different from the number of RNN
        # steps.
        def batch_labels_to_num_steps(batch_labels, lengths):
            num_steps = 0
            for labels, length in zip(batch_labels, lengths):
                num_steps += encoder_decoder.labels_to_num_steps(labels[:length])
            return np.float32(num_steps)

        num_steps = tf.py_func(
            batch_labels_to_num_steps, [labels, lengths], tf.float32)

        if mode == 'train':
            loss = tf.reduce_mean(softmax_cross_entropy)
            perplexity = tf.exp(loss)
            accuracy = tf.reduce_mean(correct_predictions)
            event_accuracy = (
                tf.reduce_sum(correct_predictions * event_positions) /
                tf.reduce_sum(event_positions))
            no_event_accuracy = (
                tf.reduce_sum(correct_predictions * no_event_positions) /
                tf.reduce_sum(no_event_positions))

            loss_per_step = tf.reduce_sum(softmax_cross_entropy) / num_steps
            perplexity_per_step = tf.exp(loss_per_step)

            optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

            train_op = tf.contrib.slim.learning.create_train_op(
                loss, optimizer, clip_gradient_norm=hparams.clip_norm)
            tf.add_to_collection('train_op', train_op)

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
                    tf.metrics.mean(softmax_cross_entropy),
                'metrics/accuracy':
                    tf.metrics.accuracy(labels_flat, predictions_flat),
                'metrics/per_class_accuracy':
                    tf.metrics.mean_per_class_accuracy(labels_flat,
                                                    predictions_flat,
                                                    num_classes),
                'metrics/event_accuracy':
                    tf.metrics.recall(event_positions, correct_predictions),
                'metrics/no_event_accuracy':
                    tf.metrics.recall(no_event_positions, correct_predictions),
                'metrics/loss_per_step':
                    tf.metrics.mean(
                        tf.reduce_sum(softmax_cross_entropy) / num_steps,
                        weights=num_steps),
            })
            for updates_op in update_ops.values():
                tf.add_to_collection('eval_ops', updates_op)

            # Perplexity is just exp(loss) and doesn't need its own update op.
            vars_to_summarize['metrics/perplexity'] = tf.exp(
                vars_to_summarize['loss'])
            vars_to_summarize['metrics/perplexity_per_step'] = tf.exp(
                vars_to_summarize['metrics/loss_per_step'])

            for var_name, var_value in six.iteritems(vars_to_summarize):
                tf.summary.scalar(var_name, var_value)
                tf.add_to_collection(var_name, var_value)

        elif mode == 'generate':
            temperature = tf.placeholder(tf.float32, [])
            if isinstance(num_classes, numbers.Number):
                softmax_flat = tf.nn.softmax(
                    tf.div(logits_flat, tf.fill([num_classes], temperature)))
                softmax = tf.reshape(
                    softmax_flat, [hparams.batch_size, -1, num_classes])
            else:
                logits_offsets = np.cumsum([0] + num_classes)
                softmax = []
                for i in range(len(num_classes)):
                    sm = tf.nn.softmax(
                        tf.div(
                            logits_flat[:, logits_offsets[i]:logits_offsets[i + 1]],
                            tf.fill([num_classes[i]], temperature)))
                    sm = tf.reshape(sm, [hparams.batch_size, -1, num_classes[i]])
                    softmax.append(sm)

            tf.add_to_collection('inputs', inputs)
            tf.add_to_collection('temperature', temperature)
            tf.add_to_collection('softmax', softmax)
            # Flatten state tuples for metagraph compatibility.
            for state in tf_nest.flatten(initial_state):
                tf.add_to_collection('initial_state', state)
            for state in tf_nest.flatten(final_state):
                tf.add_to_collection('final_state', state)

    return build
