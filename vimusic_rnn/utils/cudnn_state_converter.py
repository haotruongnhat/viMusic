import tensorflow as tf
from tensorflow.contrib import rnn as contrib_rnn

def cudnn_lstm_state_to_state_tuples(cudnn_lstm_state):
    """Convert CudnnLSTM format to LSTMStateTuples."""
    h, c = cudnn_lstm_state
    return tuple(
        contrib_rnn.LSTMStateTuple(h=h_i, c=c_i)
        for h_i, c_i in zip(tf.unstack(h), tf.unstack(c)))

def state_tuples_to_cudnn_lstm_state(lstm_state_tuples):
    """Convert LSTMStateTuples to CudnnLSTM format."""
    h = tf.stack([s.h for s in lstm_state_tuples])
    c = tf.stack([s.c for s in lstm_state_tuples])
    return (h, c)