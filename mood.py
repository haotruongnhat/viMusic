# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import re
import numpy as np
import spacy
import nltk
import unicodedata


# %%
CONTRACTION_MAP = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"}


# %%
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
nltk.download('stopwords')
nltk.download('punkt')
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


# %%
input_string = "Radio, video \nBoogie with a suitcase \nYou're living in a disco \nForget about the rat race\n\n\nLet's do the milkshake \nSelling like a hotcake \nTry some, buy some\nFee-fi-fo-fum\nTalk about pop muzik\nTalk about pop muzik"


# %%
def lyrics_prep(input_text, remove_digits=False):
    #Remove Text inside parenthesis or brackets
    processed_text = re.sub(r"\(.*?\)", "", input_text)
    processed_text = re.sub(r"\[.*?\]", "", processed_text)
    #Remove accented characters
    processed_text = unicodedata.normalize('NFKD', processed_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    #Expand contractions
    contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),
                                     flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTION_MAP.get(match)                                if CONTRACTION_MAP.get(match)                                else CONTRACTION_MAP.get(match.lower())
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
    tokens = [token for token in tokens if not token in stopword_list]
    processed_text = ' '.join(tokens)
    #Lemmatize String
    s = [token.lemma_ for token in nlp(processed_text)]
    processed_text = ' '.join(s)
    return processed_text


# %%
lyrics_prep(input_string)

# %% [markdown]
# ### Create Word Embeddings

# %%
import tensorflow_hub as hub
import tensorflow as tf
# elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)
elmo = hub.Module("3/", trainable=False)


# %%
def elmo_vectors(x):
    embeddings = elmo([x], signature='default', as_dict=True)["elmo"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(tf.math.reduce_mean(embeddings, 1))


# %%
lyrics_elmo_vectors = elmo_vectors(lyrics_prep(input_string))
lyrics_elmo_vectors

# %% [markdown]
# ### Load model

# %%
def build_model():
    input_array = tf.keras.layers.Input(shape=(1024,), dtype='float')
    dense = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_array)
    dense = tf.keras.layers.Dense(128, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(dense)
    out = tf.keras.layers.Dense(1)(dense)
    model = tf.keras.Model(inputs=input_array, outputs=out)
    model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mean_absolute_error'])
    return model
model = build_model()


# %%
mood_target = 'speechiness'


# %%
with tf.Session() as session:
    model.l
    mood_target))
    print(model.predict(lyrics_elmo_vectors))


