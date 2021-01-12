from server.workers.mood_analysis_worker import MoodAnalysisWorker
from server.workers.primer_suggestion_worker import PrimerSuggestionWorker
from server.workers.lyrics_extraction_worker import LyricsExtractionWorker
from server.workers.remi_and_mastering_worker import RemiAndMasteringWorker

def main():
    mood = MoodAnalysisWorker(
        elmo_path="models/spotify_mood_model/elmo",
        spotify_mood_path="models/spotify_mood_model/checkpoints",
        allow_memory=5000
    )

    pop_lyrics_extraction = LyricsExtractionWorker(
        model = ("pop","./models/lyrics_analysis/checkpoints/pop/20_05_2020_04-51-15.pt"),
        allow_memory=2000
    )

    primer = PrimerSuggestionWorker()

    pop_melody_generation = RemiAndMasteringWorker(
        model = ("pop","/home/vimusic/ViMusic/models/remi/checkpoint/miscellanous_one_layer/remi_models_data_v2/20200529-031105/model.pt"),
        allow_memory=9000
    )

    #initialize models
    mood.doInit()
    pop_lyrics_extraction.doInit()
    primer.doInit()
    pop_melody_generation.doInit()

    lyric = """There comes a time\n\
When we heed a certain call\n\
When the world must come together as one\n\
There are people dying"""

    example_query = {
        'status': 'wait',
        'idProcess': 'test_id',
        'percentage': 0,
        'lyric': lyric,
        'path': '.cache/.result',
        'result': {},
        'time_signature_numerator': 4,
        'time_signature_denominator': 4,
        'key_signature': 'C',
        'bpm': 'medium',
        'danceability': 1.0,
        'energy': 0.5,
        'speechiness': 0.0,
        'valence': 0.45,
        'acousticness': 0.2,
        'instrumentalness': 0.5,
        'liveness': 0.9,
        'option' : 3,
    }

    query = mood.doTask(example_query)
    query = pop_lyrics_extraction.doTask(query)
    query = primer.doTask(query)
    query = pop_melody_generation.doTask(query)

if __name__ == '__main__':
    main()
