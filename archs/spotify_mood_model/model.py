import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
from pathlib import Path
import copy
import spacy
import logging
import re
import unicodedata
from common.constants import COMPOSER
import nltk
import multiprocessing

class SpotifyMoodModel(object):
    def __init__(self,elmo_path,mood_path,cuda_device_index):
        """
        - Task: Determing the moods of the lyrics
        - Input:
            elmo_path: directory to elmo checkpoint
            mood_path: directory to mood checkpoint
            cuda_device_index: index of cuda device
        """

        self._elmo_path = elmo_path
        self._mood_path = mood_path

        self.nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner']) #it allocates in gpu: problem
        self.stopword_list = nltk.corpus.stopwords.words('english')
        self.stopword_list.remove('no')
        self.stopword_list.remove('not')

        #Initializing models
        self._config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
        self._config.gpu_options.allow_growth=True
        self._config.gpu_options.visible_device_list = "{}".format(cuda_device_index)
        self._sess = tf.compat.v1.InteractiveSession(config = self._config)
        self.elmo_embedding_layer = hub.Module(
            str(Path(self._elmo_path).absolute()),
            trainable=False
        )
        self._mood_detectors = {
            "danceability" : self.spotify_mood_model(),
            "energy" : self.spotify_mood_model(),
            "speechiness" : self.spotify_mood_model(),
            "valence" : self.spotify_mood_model(),
            "acousticness" : self.spotify_mood_model(),
            "instrumentalness" : self.spotify_mood_model(),
            "liveness" : self.spotify_mood_model(),
        }
            
    
    def spotify_mood_model(self):
        """
        Builder for the models
        """
        input_array = tf.keras.layers.Input(shape=(1024,), dtype='float')
        dense = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_array)
        dense = tf.keras.layers.Dense(128, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense)
        out = tf.keras.layers.Dense(1)(dense)
        model = tf.keras.Model(inputs=input_array, outputs=out)
        model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mean_absolute_error'])
        return model

    def _elmo_vector(self,mood_lyrics):
        embeddings = self.elmo_embedding_layer([mood_lyrics], signature='default', as_dict=True)["elmo"]
        self._sess.run(tf.compat.v1.global_variables_initializer())
        self._sess.run(tf.compat.v1.tables_initializer())
        result = self._sess.run(embeddings)
        elmo_vec = self._sess.run(tf.math.reduce_mean(result, 1))
        return elmo_vec

    def predict(self,mood_lyrics):
        """
        Predict mood of the given lyrics
            - Input:
                mood_lyrics: str
            - Output:
                a dictionary of moods
        """
        #FIXME: Different result every time. Pls help, JR :(
        ###Get mood from lyrics
        mood_lyrics = self.lyrics_prep(mood_lyrics)
        try:
            elmo_vec = self._elmo_vector(mood_lyrics)
            #predict mood for model
            mood_value = dict()
            for mood in self._mood_detectors:
                #FIXME: This needs to be loaded one time only. Its weird when the models do not work properly when loading on __init__
                self._mood_detectors[mood].load_weights(os.path.join(self._mood_path,'model_weights_{}_v1.h5'.format(mood)))
                mood_value[mood] = self._mood_detectors[mood].predict(elmo_vec)[0][0] #2d array as output
        except:
            print("Lyrics too short...")
            print(mood_lyrics)
            mood_value = dict()
            for mood in self._mood_detectors:
                mood_value[mood] = 0.0

        return mood_value

    def lyrics_prep(self,input_text,remove_digits=False):
        """
        Lyrics prepration for the model
        """
        #Remove Text inside parenthesis or brackets
        processed_text = re.sub(r"\(.*?\)", "", input_text)
        processed_text = re.sub(r"\[.*?\]", "", processed_text)

        #Remove accented characters
        processed_text = unicodedata.normalize('NFKD', processed_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        #Expand contractions
        contractions_pattern = re.compile('({})'.format('|'.join(COMPOSER.CONTRACTION_MAP.keys())),
                                        flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = COMPOSER.CONTRACTION_MAP.get(match)\
                                    if COMPOSER.CONTRACTION_MAP.get(match)\
                                    else COMPOSER.CONTRACTION_MAP.get(match.lower())
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction
        processed_text = contractions_pattern.sub(expand_match, processed_text)
        processed_text = re.sub("'", "", processed_text)
        
        #Remove Special Characters
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        processed_text = re.sub(pattern, '', processed_text)
        
        #Remove extra spaces
        processed_text = ' '.join(processed_text.split())
        
        #Remove stopwords
        tokens = nltk.tokenize.word_tokenize(processed_text)
        tokens = [token for token in tokens if not token in self.stopword_list]
        processed_text = ' '.join(tokens)
        
        #Lemmatize String
        s = [token.lemma_ for token in self.nlp(processed_text)]
        processed_text = ' '.join(s)
        return processed_text

    def close(self):
        self._sess.close()


if __name__ == '__main__':
    mood_lyrics = "\
Radio, video \n\
Boogie with a suitcase \n\
You're living in a disco \n\
Forget about the rat race\n\
\n\n\
Let's do the milkshake \n\
Selling like a hotcake \n\
Try some, buy some\n\
Fee-fi-fo-fum\n\
Talk about pop muzik\n\
Talk about pop muzik"
    sp = SpotifyMoodModel("models/spotify_mood_model/elmo","models/spotify_mood_model/checkpoints",6)
    for i in range(10):
        print(sp.predict(mood_lyrics))
    import pdb; pdb.set_trace()
    sp.close()