
import os, sys, glob
from mpipe import Stage, Pipeline
import subprocess
import nltk
import logging
import multiprocessing
import json
# Workers
from server.workers.mood_analysis_worker import MoodAnalysisWorker
from server.workers.primer_suggestion_worker import PrimerSuggestionWorker
from server.workers.lyrics_extraction_worker import LyricsExtractionWorker
from server.workers.remi_and_mastering_worker import RemiAndMasteringWorker

#Pipeline
from server.redis_queue_processor import ViQueue
import redis



def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)


def main():
    """
    Clean cache folder
    """
    files = glob.glob('.cache/.primer/*')
    for f in files:
        os.remove(f)
    """
    Declare global lock
    """
    global_lock = multiprocessing.Lock()
    pool = redis.ConnectionPool(host = '0.0.0.0',port = 6378, db = 0)

    """
    Declare AI pipeline
    """
    #1
    mood_stage = Stage(MoodAnalysisWorker,1,
        elmo_path="models/spotify_mood_model/elmo",
        spotify_mood_path="models/spotify_mood_model/checkpoints",
        allow_memory=5000,
        global_lock=global_lock,
        pool=pool
    ) #5

    #2
    pop_lyrics_extraction_stage = Stage(LyricsExtractionWorker,1,
        model = ("pop","./models/lyrics_analysis/checkpoints/pop/20_05_2020_04-51-15.pt"),
        allow_memory=2000,
        global_lock=global_lock,
        pool=pool
    ) #10
    rock_lyrics_extraction_stage = Stage(LyricsExtractionWorker,1,
        model = ("rock","./models/lyrics_analysis/checkpoints/rock/20_05_2020_04-47-07.pt"),
        allow_memory=2000,
        global_lock=global_lock,
        pool=pool
    ) #10

    electro_lyrics_extraction_stage = Stage(LyricsExtractionWorker,1,
        model = ("electro","./models/lyrics_analysis/checkpoints/rock/20_05_2020_04-47-07.pt"),
        allow_memory=2000,
        global_lock=global_lock,
        pool=pool
    ) #10

    #3
    primer_stage = Stage(PrimerSuggestionWorker,1,
        global_lock=global_lock,
        pool=pool
    ) #5

    #4
    pop_melody_generation_stage = Stage(RemiAndMasteringWorker,1,
        model = ("pop","/home/vimusic/ViMusic/models/remi/checkpoint/miscellanous_one_layer/remi_models_data_v2/20200529-031105/model.pt"),
        allow_memory=5000,
        global_lock=global_lock,
        pool=pool
    ) #20

    rock_melody_generation_stage = Stage(RemiAndMasteringWorker,1,
        model = ("rock","/home/vimusic/ViMusic/models/remi/checkpoint/miscellanous_one_layer/remi_models_data_v2/20200529-031105/model.pt"),
        global_lock=global_lock,
        pool=pool,
        allow_memory=5000
    ) #20
    electro_melody_generation_stage = Stage(RemiAndMasteringWorker,1,
        model = ("electro","/home/vimusic/ViMusic/models/remi/checkpoint/miscellanous_one_layer/remi_models_data_v2/20200529-031105/model.pt"),
        allow_memory=5000,
        global_lock=global_lock,
        pool=pool
    ) #20

    #linking stage
    mood_stage.link(pop_lyrics_extraction_stage)

    pop_lyrics_extraction_stage.link(rock_lyrics_extraction_stage)

    rock_lyrics_extraction_stage.link(electro_lyrics_extraction_stage)

    electro_lyrics_extraction_stage.link(primer_stage)

    primer_stage.link(pop_melody_generation_stage)

    pop_melody_generation_stage.link(rock_melody_generation_stage)

    rock_melody_generation_stage.link(electro_melody_generation_stage)

    # MoodAnalysisWorker
    mood_pipe = Pipeline(mood_stage)
    queue = ViQueue(mood_pipe,global_lock,pool)

    queue.start()


if __name__ == '__main__':
    # Make sure that these packages are available for workers
    # nltk and spacy
    # initialize some natural language processing libs

    subprocess.run('bash -c "source activate vimusic; spacy download en_core_web_lg"',shell=True) #load spacy model
    nltk.download('stopwords')
    nltk.download('punkt')

    #Set logging to ignore Tensorflow's log 
    set_tf_loglevel(logging.FATAL)

    main()


