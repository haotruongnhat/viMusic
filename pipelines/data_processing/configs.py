from torchvision import transforms, utils
from common.object import HParams
import pipelines.data_processing.pipelines as pipelines
import collections
from pathlib import Path
from common.object import Configs
from common.constants import COMPOSER

from common.music_item import RemiItem, MusicAutobotItem, LyricsItem
from common.vocab import RemiVocabItem, RemiMidiVocabItem, MusicAutobotVocabItem
from common.vocab_definition import REMI, MUSIC_AUTOBOT

POP_W_SIZE = 2
POP_MAX_SSM_SIZE = 50
pop_lyrics_analysis_config = Configs(
    name='pop_lyrics_segmentation',
    hparams=HParams(
        batch_size = 10,
        epoch=30,
        learning_rate=0.001,
        dropout=0.2,
        checkpoint_root_path = "./models/lyrics_analysis/checkpoints"
    ),
    handler=LyricsItem, #containing a segment of item
    dataset_dir='models/lyrics_analysis/datasets',
    data_path='models/lyrics_analysis/datasets/wasabi_pop.pkl',
    max_parts=18,
    max_songs=600,
    shuffle=False,
    train_shuffle=True,
    transform_pipeline = transforms.Compose([
        pipelines.LyricsSegment(
        w_size=POP_W_SIZE, #default
        min_ssm_size=10,
        max_ssm_size=POP_MAX_SSM_SIZE
        ) #split song into multiple LyricsSegmentItem
    ]),
    min_ssm_size=10,
    max_ssm_size=POP_MAX_SSM_SIZE,
    w_size=POP_W_SIZE,
    output_size=POP_MAX_SSM_SIZE,
    num_workers=4,
    genre=COMPOSER.POP,
    ratio = 0.8
)

ROCK_W_SIZE = 2
ROCK_MAX_SSM_SIZE = 50
rock_lyrics_analysis_config = Configs(
    name='rock_lyrics_segmentation',
    hparams=HParams(
        batch_size = 10,
        epoch=70,
        learning_rate=0.001,
        dropout=0.1,
        checkpoint_root_path = "./models/lyrics_analysis/checkpoints"
    ),
    handler=LyricsItem, #containing a segment of item
    dataset_dir='models/lyrics_analysis/datasets',
    data_path='models/lyrics_analysis/datasets/wasabi_rock.pkl',
    max_parts=18,
    max_songs=600,
    shuffle=False,
    train_shuffle=True,
    transform_pipeline = transforms.Compose([
        pipelines.LyricsSegment(
        w_size=ROCK_W_SIZE, #default
        min_ssm_size=10,
        max_ssm_size=ROCK_MAX_SSM_SIZE
        ) #split song into multiple LyricsSegmentItem
    ]),
    min_ssm_size=10,
    max_ssm_size=ROCK_MAX_SSM_SIZE,
    w_size=ROCK_W_SIZE,
    output_size=ROCK_MAX_SSM_SIZE,
    num_workers=4,
    genre=COMPOSER.ROCK,
    ratio = 0.8
)

ELECTRO_W_SIZE = 2
ELECTRO_MAX_SSM_SIZE = 50
electro_lyrics_analysis_config = Configs(
    name='electro_lyrics_segmentation',
    hparams=HParams(
        batch_size = 10,
        epoch=70,
        learning_rate=0.001,
        dropout=0.1,
        checkpoint_root_path = "./models/lyrics_analysis/checkpoints"
    ),
    handler=LyricsItem, #containing a segment of item
    dataset_dir='models/lyrics_analysis/datasets',
    data_path='models/lyrics_analysis/datasets/wasabi_electro.pkl',
    max_parts=18,
    max_songs=600,
    shuffle=False,
    train_shuffle=True,
    transform_pipeline = transforms.Compose([
        pipelines.LyricsSegment(
        w_size=ELECTRO_W_SIZE, #default
        min_ssm_size=10,
        max_ssm_size=ELECTRO_MAX_SSM_SIZE
        ) #split song into multiple LyricsSegmentItem
    ]),
    min_ssm_size=10,
    max_ssm_size=ELECTRO_MAX_SSM_SIZE,
    w_size=ELECTRO_W_SIZE,
    output_size=ELECTRO_MAX_SSM_SIZE,
    num_workers=4,
    genre=COMPOSER.ELECTRO,
    ratio = 0.8
)

remi_no_sharp_config = Configs(
    name= 'no_sharp',
    hparams=HParams(),
    handler= RemiItem, # MusicItem handler
    vocab = RemiVocabItem(REMI.INDEX_TOKENS),
    transform_pipeline= transforms.Compose([
        pipelines.IsTimeSignature44(),
        pipelines.IsValidMelody(min_notes_per_bar = 2.),
        pipelines.DataCompression(),
        pipelines.Tokenizing(padding_len=0)
    ]
    ),
    seq_len=200,
    file_extensions = ['xml','mxl','midi','mid'],
    dataset_dir = 'models/remi/dataset_v2',
    data_path = 'models/remi/dataset_v2_pkl/dataset_no_sharp_seq_len_200.pkl',
    num_workers=4
)

# remi_no_sharp_with_syllable_config = Configs(
#     name= 'no_sharp_with_syllable',
#     hparams=HParams(
#     ),
#     handler= RemiItem, # MusicItem handler
#     vocab = RemiVocabItem(REMI.INDEX_TOKENS),
#     syll_vec_length=10,
#     transform_pipeline= transforms.Compose([
#         pipelines.IsTimeSignature44(),
#         pipelines.IsValidMelody(min_notes_per_bar = 2.),
#         pipelines.DataCompression()
#     ]
#     ),
#     file_extensions = ['xml','mxl'],
#     dataset_dir = 'models/remi/processed_voice',
#     data_path='models/remi/dataset_v2_pkl/voice_no_sharp_with_syllable.pkl',
#     num_workers=4
# )
