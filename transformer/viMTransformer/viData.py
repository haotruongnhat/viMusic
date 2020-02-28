import utils
import random
import pickle
from tensorflow.python import keras
from tensorflow.train import SequenceExample
import numpy as np
import params as par


class Data:
    def __init__(self, dir_path):
        self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
        self.file_dict = {
            'train': self.files[:int(len(self.files) * 0.8)],
            'eval': self.files[int(len(self.files) * 0.8): int(len(self.files) * 0.9)],
            'test': self.files[int(len(self.files) * 0.9):],
        }
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        pass

    def __repr__(self):
        return '<class Data has "'+str(len(self.files))+'" files>'

    def batch(self, batch_size, length, mode='train'):

        batch_files = random.sample(self.file_dict[mode], k=batch_size)
        # set_trace()
        batch_data = [
            self._get_seq(file, length)
            for file in batch_files
        ]
        return np.array(batch_data)  # batch_size, seq_len

    def seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length]
        y = data[:, length:]
        return x, y

    def smallest_encoder_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length//100]
        y = data[:, length//100:length//100+length]
        return x, y

    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def random_sequential_batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = []
        for i in range(batch_size):
            data = self._get_seq(batch_files[i])
            for j in range(len(data) - length):
                batch_data.append(data[j:j+length])
                if len(batch_data) == batch_size:
                    return batch_data

    def sequential_batch(self, batch_size, length):
        batch_data = []
        data = self._get_seq(self.files[self._seq_file_name_idx])

        while len(batch_data) < batch_size:
            while self._seq_idx < len(data) - length:
                batch_data.append(data[self._seq_idx: self._seq_idx + length])
                self._seq_idx += 1
                if len(batch_data) == batch_size:
                    return batch_data

            self._seq_idx = 0
            self._seq_file_name_idx = self._seq_file_name_idx + 1
            if self._seq_file_name_idx == len(self.files):
                self._seq_file_name_idx = 0
                print('iter intialized')

    def _get_seq(self, fname, max_length=None):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                data = np.append(data, par.token_eos)
                while len(data) < max_length:
                    data = np.append(data, par.pad_token)
        return data


class PositionalY:
    def __init__(self, data, idx):
        self.data = data
        self.idx = idx

    def position(self):
        return self.idx

    def data(self):
        return self.data

    def __repr__(self):
        return '<Label located in {} position.>'.format(self.idx)


def add_noise(inputs: np.array, rate:float = 0.01): # input's dim is 2
    seq_length = np.shape(inputs)[-1]

    num_mask = int(rate * seq_length)
    for inp in inputs:
        rand_idx = random.sample(range(seq_length), num_mask)
        inp[rand_idx] = random.randrange(0, par.pad_token)

    return inputs


class viData(Data):
    def __init__(self, dir_path):
        self.files = list(utils.find_files_by_extensions(dir_path, ['.tfrecord']))
        # set_trace()
        self.length_data, self.dataset = self._read_tf_record_files(self.files, input_size=11)
        set_trace()
        self.file_dict = {
            'train': self.dataset[:int(self.length_data * 0.8)],
            'eval': self.dataset[int(self.length_data * 0.8): int(self.length_data * 0.9)],
            'test': self.dataset[int(self.length_data * 0.9):],
        }
        self._seq_file_name_idx = 0
        self._seq_idx = 0
    
    def __repr__(self):
        return '<class Data has "'+str(self.length_data)+'" files>'
    
    def batch(self, batch_size, length, mode='train'):

        batch_token = []
        batch_files = random.sample(self.file_dict[mode], k=batch_size)
        # set_trace()
        batch_token = [data[1] for data in batch_files]
        # set_trace()
        batch_data = [
            self._get_seq(seq, length)
            for seq in batch_token
        ]
        return np.array(batch_data)

    def _get_seq(self, data, max_length):
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                # data = np.append(data, par.token_eos)
                # while len(data) < max_length:
                #     data = np.append(data, par.pad_token)
                data = None
        return data
                
    def seq2seq_batch(self, batch_size, length, mode='train'):
        # set_trace()
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length]
        y = data[:, length:]
        return x, y

    def slide_seq2seq_batch(self, batch_size, length, mode='train'):
        data = self.batch(batch_size, length+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def _read_tf_record_files(self, file_list, input_size=None, label_shape=None,
                                shuffle=False):
        file_queue =tf.data.Dataset.from_tensor_slices(file_list)
        tfrecord_dataset = tf.data.TFRecordDataset(file_queue)
        
        def read_tfrecord(serialized_example):
            sequence_features = {
                'inputs': tf.io.FixedLenSequenceFeature(shape=[input_size],
                                                dtype=tf.float32),
                'labels': tf.io.FixedLenSequenceFeature(shape=[],
                                                dtype=tf.int64)
            }
            # set_trace()
            _, example = tf.io.parse_single_sequence_example(
                        serialized_example, sequence_features = sequence_features)

            return example
        parsed_dataset = tfrecord_dataset.map(read_tfrecord)
        input_tensors = []
        for seq in parsed_dataset:
            length = tf.shape(seq['inputs'])[0]
            input_tensors.append([seq['inputs'], seq['labels'], length])
        
        return len(input_tensors), input_tensors


if __name__ == '__main__':
    import pprint
    # tf.executing_eagerly()
    from tensorflow.python.framework.ops import disable_eager_execution
    import tensorflow.compat.v1 as tf1
    from pdb import set_trace
    # import magenta_v2
    # disable_eager_execution()
    file_path = '/home/tony/Vimusic/test_data'
    # filenames = ['/home/tony/Vimusic/retrain_data/performance/training_performances.tfrecord']
    # file_queue =tf.data.Dataset.from_tensor_slices(filenames)
    # tfrecord_dataset = tf.data.TFRecordDataset(file_path)
    dataset = viData(file_path)
    batch_x, batch_y = dataset.seq2seq_batch(2, 128)
    # fname = '/home/tony/Vimusic/test_data/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_02_WAV.midi.pickle'
    # with open(fname, 'rb') as f:
    #     data = pickle.load(f)

    # print(data)
    # batch = dataset.batch(2, 10)
    # print(batch)
    # def read_tfrecord(serialized_example):
    #     sequence_features = {
    #         'inputs': tf.io.FixedLenSequenceFeature(shape=[11],
    #                                         dtype=tf.float32),
    #         'labels': tf.io.FixedLenSequenceFeature(shape=[],
    #                                         dtype=tf.int64)
    #     }
    #     # set_trace()
    #     _, example = tf.io.parse_single_sequence_example(
    #                 serialized_example, sequence_features = sequence_features)
        
    
    #     return example
    
    # parsed_dataset = tfrecord_dataset.map(read_tfrecord)
    # print(type(parsed_dataset))
    # test = parsed_dataset.batch(3)
    # print(list(test.as_numpy_iterator()))
    # # set_trace()
    # parsed_dataset = parsed_dataset.batch(2)
    # parsed_dataset = parsed_dataset.as_numpy_iterator()
    # input_tensors = []
    # for sequence in parsed_dataset:
    #     length = tf.shape(sequence['inputs'])[0]
    #     input_tensors.append([sequence['inputs'], sequence['labels'], length])
    # dataset = tf.data.Dataset.from_tensors(input_tensors)
    # tf.data.Dataset(input_tensors)
    # dataset = tf.data.Dataset.range(8)
    # dataset = dataset.batch(3) 
    # print(list(dataset.as_numpy_iterator()))
    # print(parsed_dataset['inputs'])
    # for seq in parsed_dataset.take(1):
    #     print(seq[1])
    # np_data = list(parsed_dataset.as_numpy_iterator())
    # print(type(np_data[0][1]['inputs']))

    
   
