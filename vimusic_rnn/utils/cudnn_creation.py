import tensorflow as tf
from tensorflow.contrib import cudnn_rnn as contrib_cudnn_rnn
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import rnn as contrib_rnn

from .cudnn_state_converter import state_tuples_to_cudnn_lstm_state, cudnn_lstm_state_to_state_tuples



def make_cudnn(inputs, rnn_layer_sizes, batch_size,\
    rnn_cell_func=contrib_cudnn_rnn.CudnnCompatibleLSTMCell,\
    residual_connections=False):
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
    
    #Transposing input
    cudnn_inputs = tf.transpose(inputs, [1, 0, 2])
    # We need to make multiple calls to CudnnLSTM, keeping the initial and final
    # states at each layer.
    initial_state = []
    final_state = []

    #due to residual connection, multiRNNCell has to be separated!
    for i in range(len(rnn_layer_sizes)):
        # If we're using residual connections and this layer is not the same size
        # as the previous layer, we need to project into the new size so the
        # (projected) input can be added to the output.
        if residual_connections:
            if i == 0 or rnn_layer_sizes[i] != rnn_layer_sizes[i - 1]:
                cudnn_inputs = contrib_layers.linear(cudnn_inputs, rnn_layer_sizes[i])
        #input and output size is the same
        layer_initial_state = (contrib_rnn.LSTMStateTuple(
            h=tf.zeros([batch_size, rnn_layer_sizes[i]], dtype=tf.float32),
            c=tf.zeros([batch_size, rnn_layer_sizes[i]], dtype=tf.float32)),)

        # At generation time we use CudnnCompatibleLSTMCell.
        cell = contrib_rnn.MultiRNNCell(
            [rnn_cell_func(rnn_layer_sizes[i])])
        cudnn_outputs, layer_final_state = tf.nn.dynamic_rnn(
            cell, cudnn_inputs, initial_state=layer_initial_state,
            time_major=True,
            scope='cudnn_lstm/rnn' if i == 0 else 'cudnn_lstm_%d/rnn' % i)

        if residual_connections:
            cudnn_outputs += cudnn_inputs

        cudnn_inputs = cudnn_outputs
        initial_state += layer_initial_state
        final_state += layer_final_state

    outputs = tf.transpose(cudnn_outputs, [1, 0, 2])
    return outputs, tuple(initial_state), tuple(final_state)

def make_cudnn_for_generation(inputs, rnn_layer_sizes,batch_size,\
    rnn_cell_func=contrib_cudnn_rnn.CudnnLSTM,\
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

    cudnn_inputs = tf.transpose(inputs, [1, 0, 2])
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

        cudnn_initial_state = state_tuples_to_cudnn_lstm_state(layer_initial_state)
        cell = rnn_cell_func(
            num_layers=1,
            num_units=rnn_layer_sizes[i],
            direction='unidirectional',
            dropout=1.0 - dropout_keep_prob)
        cudnn_outputs, cudnn_final_state = cell(
            cudnn_inputs, initial_state=cudnn_initial_state,
            training=False)
        layer_final_state = cudnn_lstm_state_to_state_tuples(cudnn_final_state)

        if residual_connections:
            cudnn_outputs += cudnn_inputs
            
        cudnn_inputs = cudnn_outputs
        initial_state += layer_initial_state
        final_state += layer_final_state

    outputs = tf.transpose(cudnn_outputs, [1, 0, 2])
    return outputs, tuple(initial_state), tuple(final_state)