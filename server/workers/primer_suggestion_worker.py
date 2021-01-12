from archs.primer_suggestion.primer_suggestion import PrimerSuggestion
import json
import uuid
import os

from server.workers.cuda_worker import CudaWorker

class PrimerSuggestionWorker(CudaWorker):
    """
    PrimerSuggestionWorker suggest primer melody for user according to mood
    """
    def __init__(self,global_lock=None,pool=None):
        """
        Redis
        """
        super(PrimerSuggestionWorker,self).__init__(global_lock,pool)
    
    def doInit(self):
        """
        Initialize the algorithm
        """
        self.acquire()
        self._primer_suggestion = PrimerSuggestion()
        self.release()


    def doTask(self,query):
        """
        query = {
            "time_signature_numerator" : (input,int),
            "time_signature_denominator" : (intput,int),
            "key_signature" :(input,string),
            "bpm" : string,   
            "danceability" : (input,float),
            "energy" : (input,float),
            "speechiness" : (input,float),
            "valence" : (input,float),
            "acousticness" : (input,float),
            "instrumentalness" : (input,float),
            "liveness" : (input,float),
            "status" : string,
            "idProcess" : string,
            "percentage" : int,
            "option" : int,
            "lyric" : string,
            "path" : string,
            "result" : {},
            "mood_analysis" : [],
            "{}_segmented_lyric" : matrix of string, with {} = self._genre,
            "{}_original_lyric" : matrix of string, with {} = self._genre,
            "{}_section_groups": dictionary of segments, with {} = self._genre,
            "{}_boundaries" : list of int denoting end of segments, with {} = self._genre,
            "primer_path" (output,string)
        })
        """
        if query is None: 
            self._primer_suggestion.stop()
            return None

        #Getting mood
        danceability = query['danceability']
        energy = query['energy']
        speechiness = query['speechiness']
        acousticness = query['acousticness']
        instrumentalness = query['instrumentalness']
        liveness = query['liveness']
        valence = query['valence']

        #FIXME: Waiting for front end
        tonic = 'C'
        mode = 'Major'

        #Getting signature
        numerator = query["time_signature_numerator"]
        denominator = query["time_signature_denominator"]

        #Declare primer path
        primer_file_path = os.path.abspath(os.path.join("./.cache/.primer",str(uuid.uuid4()) + ".pkl"))

        #Suggest
        self._primer_suggestion.suggest(danceability,
            energy,
            speechiness,
            acousticness,
            instrumentalness,
            liveness,
            valence,
            tonic,
            mode,
            numerator,
            denominator,
            primer_file_path)
        query["primer_path"] = primer_file_path

        #Send to redis to update progress
        query["percentage"] += 5
        self.update_progress(query)

        print("{}: LyricsExtraction done".format(query["idProcess"]))

        return query    
