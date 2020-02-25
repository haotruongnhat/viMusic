from model import MusicTransformerDecoder, MusicTransformer
<<<<<<< HEAD
=======
import os 
>>>>>>> b515b056c6a77846447518509ece1b5406b8f0d2
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import argparse
import datetime
import sys
tf.keras.backend.set_floatx('float32')
# tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float)
parser.add_argument('--batch_size', default=5, help='batch size', type=int)
parser.add_argument('--pickle_dir', default='music')
<<<<<<< HEAD
parser.add_argument('--max_seq', default=128,  type=int)
parser.add_argument('--epochs', default=100,  type=int)
parser.add_argument('--load_path', default=None,  type=str)
parser.add_argument('--save_path', default="./result")
=======
parser.add_argument('--max_seq', default=1500, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--save_path', default="/home/Projects/viMusic/tony/MT_models_TF")
>>>>>>> b515b056c6a77846447518509ece1b5406b8f0d2
parser.add_argument('--is_reuse', default=False)
parser.add_argument('--multi_gpu', default=False)
parser.add_argument('--num_layers', default=6, type=int)

args = parser.parse_args()


# set arguments
l_r = args.lr
batch_size = args.batch_size
pickle_dir = args.pickle_dir
max_seq = args.max_seq
epochs = args.epochs
is_reuse = args.is_reuse
load_path = args.load_path
save_path = args.save_path
multi_gpu = args.multi_gpu
num_layer = args.num_layers


# load data
dataset = Data(args.pickle_dir)
print(dataset)


# load model
learning_rate = callback.CustomSchedule(par.embedding_dim) if l_r is None else l_r
opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

if multi_gpu:
    strategy = tf.distribute.MirroredStrategy()
# define model
    with strategy.scope():
        mt = MusicTransformer(
                embedding_dim=256,
                vocab_size=par.vocab_size,
                num_layer=6,
                max_seq=max_seq,
                dropout=0.2,
                debug=False, loader_path=load_path)
        mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)

        # Train Start
        for e in range(epochs):
            mt.reset_metrics()
            for b in range(len(dataset.files) // batch_size):
                try:
                    batch_x, batch_y = dataset.seq2seq_batch(batch_size, max_seq)
                except:
                    continue
                result_metrics = mt.train_on_batch(batch_x, batch_y)
                if b % 100 == 0:
                    eval_x, eval_y = dataset.seq2seq_batch(batch_size, max_seq, 'eval')
                    eval_result_metrics = mt.evaluate(eval_x, eval_y)
                    mt.save(save_path)
                    print('\n====================================================')
                    print('Epoch/Batch: {}/{}'.format(e, b))
                    print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1]))
                    print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1]))
else:
# define model
<<<<<<< HEAD
=======
if multi_gpu: 
    pass
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = tf.keras.layers.Input(shape=(1,))
        predictions = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

else:
>>>>>>> b515b056c6a77846447518509ece1b5406b8f0d2
    mt = MusicTransformer(
                embedding_dim=256,
                vocab_size=par.vocab_size,
                num_layer=num_layer,
                max_seq=max_seq,
                dropout=0.2,
                debug=False, loader_path=load_path)
    mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)


<<<<<<< HEAD
    # define tensorboard writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = 'logs/mt/'+current_time+'/train'
    eval_log_dir = 'logs/mt/'+current_time+'/eval'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)


    # Train Start
    idx = 0
    print('-------------Start Training-------------')
    print(epochs)
    for e in range(epochs):
        mt.reset_metrics()
        for b in range(len(dataset.files) // batch_size):
            try:
                batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
            except:
                continue
            result_metrics = mt.train_on_batch(batch_x, batch_y)
            if b % 100 == 0:
                eval_x, eval_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval')
                eval_result_metrics, weights = mt.evaluate(eval_x, eval_y)
                mt.save(save_path)
                with train_summary_writer.as_default():
                    if b == 0:
                        tf.summary.histogram("target_analysis", batch_y, step=e)
                        tf.summary.histogram("source_analysis", batch_x, step=e)

                    tf.summary.scalar('loss', result_metrics[0], step=idx)
                    tf.summary.scalar('accuracy', result_metrics[1], step=idx)

                with eval_summary_writer.as_default():
                    if b == 0:
                        mt.sanity_check(eval_x, eval_y, step=e)

                    tf.summary.scalar('loss', eval_result_metrics[0], step=idx)
                    tf.summary.scalar('accuracy', eval_result_metrics[1], step=idx)
                    for i, weight in enumerate(weights):
                        with tf.name_scope("layer_%d" % i):
                            with tf.name_scope("w"):
                                utils.attention_image_summary(weight, step=idx)
                    # for i, weight in enumerate(weights):
                    #     with tf.name_scope("layer_%d" % i):
                    #         with tf.name_scope("_w0"):
                    #             utils.attention_image_summary(weight[0])
                    #         with tf.name_scope("_w1"):
                    #             utils.attention_image_summary(weight[1])
                idx += 1
                print('\n====================================================')
                print('Epoch/Batch: {}/{}'.format(e, b))
                print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1]))
                print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1]))
=======
# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = os.path.join(args.save_path,'logs/mt/'+current_time+'/train')
eval_log_dir = os.path.join(args.save_path,'logs/mt/'+current_time+'/eval')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)


# Train Start
idx = 0
print('-------------Start Training-------------')
print(epochs)
print(len(dataset.files))
for e in range(epochs):
    mt.reset_metrics()
    for b in range(len(dataset.files) // batch_size-1):
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq)
        except:
            continue
        result_metrics = mt.train_on_batch(batch_x, batch_y)
        if b % 100 == 0:
            eval_x, eval_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval')
            eval_result_metrics, weights = mt.evaluate(eval_x, eval_y)
            mt.save(save_path)
            with train_summary_writer.as_default():
                if b == 0:
                    tf.summary.histogram("target_analysis", batch_y, step=e)
                    tf.summary.histogram("source_analysis", batch_x, step=e)

                tf.summary.scalar('loss', result_metrics[0], step=idx)
                tf.summary.scalar('accuracy', result_metrics[1], step=idx)

            with eval_summary_writer.as_default():
                if b == 0:
                    mt.sanity_check(eval_x, eval_y)

                tf.summary.scalar('loss', eval_result_metrics[0], step=idx)
                tf.summary.scalar('accuracy', eval_result_metrics[1], step=idx)
                # for i, weight in enumerate(weights):
                #     with tf.name_scope("layer_%d" % i):
                #         with tf.name_scope("w"):
                #             utils.attention_image_summary(weight, step=idx)
                for i, weight in enumerate(weights):
                    with tf.name_scope("layer_%d" % i):
                        with tf.name_scope("_w0"):
                            utils.attention_image_summary(weight[0])
                        with tf.name_scope("_w1"):
                            utils.attention_image_summary(weight[1])
            idx += 1
            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1]))
            print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1]))
>>>>>>> b515b056c6a77846447518509ece1b5406b8f0d2


