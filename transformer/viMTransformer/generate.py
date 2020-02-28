import os

from model import MusicTransformer, MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import datetime
import argparse
from midi_processor.processor import decode_midi, encode_midi


parser = argparse.ArgumentParser()

parser.add_argument('--max_seq', default=1500, help='최대 길이', type=int)
parser.add_argument('--load_path', default="result/dec0722", help='모델 로드 경로', type=str)
parser.add_argument('--primer_midi', default='/home/Projects/viMusic/tony/primer/row_row_row_PNO.mid')
parser.add_argument('--output_dir', default='/home/Projects/viMusic/tony/output_midi/')
parser.add_argument('--mode', default='enc-dec')
parser.add_argument('--beam', default=None, type=int)
parser.add_argument('--length', default=2048, type=int)
# parser.add_argument('--save_path', default='bin/generated.mid', type=str)


args = parser.parse_args()


# set arguments
primer_midi = args.primer_midi
max_seq = args.max_seq
load_path = args.load_path
mode = args.mode
beam = args.beam
length = args.length
output_dir = args.output_dir

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt/generate_'+current_time+'/generate'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
save_path = os.path.join(output_dir, 'generated_'+current_time+'.mid')
primer_name = os.path.join(output_dir, 'primer_'+current_time+'.mid')
if mode == 'enc-dec':
    print(">> generate with original seq2seq wise... beam size is {}".format(beam))
    mt = MusicTransformer(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=6,
            max_seq=max_seq,
            dropout=0.2,
            debug=False, loader_path=load_path)
else:
    print(">> generate with decoder wise... beam size is {}".format(beam))
    mt = MusicTransformerDecoder(loader_path=load_path)
np.random.seed(1000)

cutlength = 40

# import pdb; pdb.set_trace()
inputs = encode_midi(primer_midi)

start_p = np.random.randint(0, len(inputs)-cutlength-10)
# for i in inputs:
#     print(i)

with gen_summary_writer.as_default():
    # result = mt.generate(inputs[start_p:start_p+cutlength], beam=beam, length=length, tf_board=True)
    decode_midi(inputs[start_p:start_p+cutlength], file_path=primer_name)

for i in result:
    print(i)

# if mode == 'enc-dec':
#     decode_midi(list(inputs[-1*par.max_seq:]) + list(result[1:]), file_path=save_path)
# else:
#     decode_midi(result, file_path=save_path)
