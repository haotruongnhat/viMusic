from .utils import get_graph_builder
from .utils import gen_tf_dataset_from_tfrecord, gen_tf_dataset

gen_tf_dataset('pop_dataset')
gen_tf_dataset_from_tfrecord('pop_dataset')
