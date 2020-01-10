from magenta.music.encoder_decoder import make_sequence_example

class ConditionalInConditionalEventSequenceEncoderDecoder(object):
    def __init__(self, control_encoder_decoder, target_encoder_decoder):
        """Initialize a ConditionalEventSequenceEncoderDecoder object.

        Args:
        control_encoder_decoder: The EventSequenceEncoderDecoder to encode/decode
            the control sequence.
        target_encoder_decoder: The EventSequenceEncoderDecoder to encode/decode
            the target sequence.
        """
        self._control_encoder_decoder = control_encoder_decoder
        self._target_encoder_decoder = target_encoder_decoder #conditional encoder decoder

    @property
    def input_size(self):
        """The size of the concatenated control and target input vectors.

        Returns:
            An integer, the size of an input vector.
        """
        return (self._control_encoder_decoder.input_size +
                self._target_encoder_decoder.input_size)

    @property
    def num_classes(self):
        """The range of target labels used by this model.

        Returns:
            An integer, the range of integers that can be returned by
                self.events_to_label.
        """
        return self._target_encoder_decoder.num_classes

    @property
    def default_event_label(self):
        """The class label that represents a default target event.

        Returns:
        An integer, the class label that represents a default target event.
        """
        return self._target_encoder_decoder.default_event_label

    def events_to_input(self, control_events, target_control_events, target_target_events, position):
        return (
            self._control_encoder_decoder.events_to_input(
                control_events, position + 1) +
            self._target_encoder_decoder.events_to_input(target_control_events, target_target_events, position))

    def events_to_label(self, target_events, position):
        """Returns the label for the given position in the target event sequence.

        Args:
        target_events: A list-like sequence of target events.
        position: An integer event position in the target event sequence.

        Returns:
        A label, an integer.
        """
        return self._target_encoder_decoder.events_to_label(target_events, position)

    def class_index_to_event(self, class_index, target_events):
        """Returns the event for the given class index.

        This is the reverse process of the self.events_to_label method.

        Args:
        class_index: An integer in the range [0, self.num_classes).
        target_events: A list-like sequence of target events.

        Returns:
        A target event value.
        """
        return self._target_encoder_decoder.class_index_to_event(
            class_index, target_events)

    def labels_to_num_steps(self, labels):
        """Returns the total number of time steps for a sequence of class labels.

        Args:
        labels: A list-like sequence of integers in the range
            [0, self.num_classes).

        Returns:
        The total number of time steps for the label sequence, as determined by
        the target encoder/decoder.
        """
        return self._target_encoder_decoder.labels_to_num_steps(labels)

    def encode(self, control_events, target_control_events, target_target_events):
        if len(control_events) != len(target_control_events) or \
        len(control_events) != len(target_target_events) or \
        len(target_control_events) != len(target_target_events):
            raise ValueError('must have the same number of events')

        inputs = []
        labels = []
        for i in range(len(target_target_events) - 1):
            inputs.append(self.events_to_input(control_events, target_control_events, target_target_events, i))
            labels.append(self.events_to_label(target_target_events, i + 1))
        return make_sequence_example(inputs, labels)

    def get_inputs_batch(self, control_event_sequences, target_control_event_sequences,
    target_target_event_sequences, full_length=False):
        if len(control_event_sequences) != len(target_control_event_sequences) or \
        len(control_event_sequences) != len(target_target_event_sequences) or \
        len(target_control_event_sequences) != len(target_target_event_sequences):
            raise ValueError('must have the same number of events')

        inputs_batch = []
        for control_events, target_control_events, target_target_events in zip(
        control_event_sequences, target_control_event_sequences,target_target_event_sequences):
            if len(control_events) < len(target_control_events):
                raise ValueError('control event sequence must be longer than control target '
                                'event sequence (%d control events but %d control target '
                                'events)' % (len(control_events), len(target_control_events)))
            if len(target_control_events) < len(target_target_events):
                raise ValueError('control target event sequence must be longer than target target '
                                'event sequence (%d control target events but %d target target '
                                'events)' % (len(target_control_events), len(target_target_events)))
            inputs = []
            if full_length:
                for i in range(len(target_target_events) - 1):
                    inputs.append(self.events_to_input(control_events, target_control_events, target_target_events, i))
            else:
                inputs.append(self.events_to_input(
                control_events, target_control_events, target_target_events, len(target_target_events) - 1))
            inputs_batch.append(inputs)
        return inputs_batch

    def evaluate_log_likelihood(self, target_event_sequences, softmax):
        """Evaluate the log likelihood of multiple target event sequences.

        Args:
        target_event_sequences: A list of target EventSequence objects.
        softmax: A list of softmax probability vectors. The list of softmaxes
            should be the same length as the list of target event sequences. The
            softmax vectors are assumed to have been generated by a full-length
            inputs batch.

        Returns:
        A Python list containing the log likelihood of each target event sequence.
        """
        return self._target_encoder_decoder.evaluate_log_likelihood(
            target_event_sequences, softmax)

    def extend_event_sequences(self, target_event_sequences, softmax):
        """Extends the event sequences by sampling the softmax probabilities.

        Args:
        target_event_sequences: A list of target EventSequence objects.
        softmax: A list of softmax probability vectors. The list of softmaxes
            should be the same length as the list of event sequences.

        Returns:
        A Python list of chosen class indices, one for each target event sequence.
        """
        return self._target_encoder_decoder.extend_event_sequences(
            target_event_sequences, softmax)