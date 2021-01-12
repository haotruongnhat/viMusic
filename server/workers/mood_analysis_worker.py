import os
import json

from archs.spotify_mood_model.model import SpotifyMoodModel

from server.workers.cuda_worker import CudaWorker

class MoodAnalysisWorker(CudaWorker):
    """
    MoodAnalysisWorker extract mood from the lyrics
    """
    def __init__(self,
        elmo_path,
        spotify_mood_path,
        allow_memory,
        global_lock=None,
        pool=None,
        ):

        super(MoodAnalysisWorker,self).__init__(global_lock,pool,allow_memory)

        self._elmo_checkpoint_path = os.path.abspath(elmo_path)
        self._spotify_mood_path = os.path.abspath(spotify_mood_path)

        self._mood_list = ["danceability","energy","speechiness","valence","acousticness",\
        "instrumentalness","liveness"]

    def doInit(self):
        """
        Initializing the model
        """
        self.acquire()
        self.init_cuda_memory()
        print("A MoodAnalysisWorker process is using GPU {}".format(self._cuda_device_index))
        self._spotify_model = SpotifyMoodModel(
            self._elmo_checkpoint_path,
            self._spotify_mood_path,self._cuda_device_index)
        self.release()

    def doTask(self,query):
        """
        query = {
            "time_signature_numerator" : int,
            "time_signature_denominator" : int,
            "key_signature" : string,
            "bpm" : string,   
            "danceability" : (output,float),
            "energy" : (output,float),
            "speechiness" : (output,float),
            "valence" : (output,float),
            "acousticness" : (output,float),
            "instrumentalness" : (output,float),
            "liveness" : (output,float),
            "status" : string,
            "idProcess" : string,
            "percentage" : (int),
            "option" : int,
            "lyric" : (input, string),
            "path" : string,
            "result" : {},
        })
        """
        if query is None: #End of every queue, shut down
            self._spotify_model.close()
            return None

        
        lyrics = query['lyric']
        mood_result = self._spotify_model.predict(lyrics)
        for mood in mood_result:
            query[mood] = mood_result[mood]

        #Send to redis to update progress
        query['percentage'] += 5
        self.update_progress(query)

        print("{}: MoodAnalysis done".format(query["idProcess"]))

        return query
