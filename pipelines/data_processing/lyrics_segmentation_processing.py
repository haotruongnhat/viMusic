from pipelines.data_processing.transform import LyricsDatasetTransform
from pipelines.data_processing.configs import pop_lyrics_analysis_config,rock_lyrics_analysis_config, electro_lyrics_analysis_config
import argparse

parser = argparse.ArgumentParser(description='Processor for lyricsc segmentation')
parser.add_argument('--genre',type=str,default='pop')

def main():
    args = parser.parse_args()
    if args.genre == 'pop':
        lyrics_dataset_transform = LyricsDatasetTransform(pop_lyrics_analysis_config)
    elif args.genre == 'rock':
        lyrics_dataset_transform = LyricsDatasetTransform(rock_lyrics_analysis_config)
    elif args.genre == 'electro':
        lyrics_dataset_transform = LyricsDatasetTransform(electro_lyrics_analysis_config)
    lyrics_dataset_transform.save_preprocess_dataset()

if __name__ == '__main__':
    main()