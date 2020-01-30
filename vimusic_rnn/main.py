#from .utils import get_graph_builder
from .utils import gen_tf_dataset_from_tfrecord, gen_tf_dataset

from vimusic_rnn.progress import train_vimusic_rnn
from vimusic_rnn.progress import generate_melody


#gen_tf_dataset('pop_dataset')
#gen_tf_dataset_from_tfrecord('pop_dataset')

#train_vimusic_rnn('pop_dataset',training_steps = 10)
generate_melody('pop_dataset', duration_in_seconds=15,temperature=1.0, steps_per_iteration=1,beam_size=1, branch_factor=1)
