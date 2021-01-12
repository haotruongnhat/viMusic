import pandas as pd
import os, sys
from scipy.spatial import distance
import numpy as np
import math
import ast
import random

class PrimerSuggest(object):
    def __init__(self):
        pass

    
#extract mood
genre = sys.argv[1]

mood_value = sys.argv[2:]
mood_name = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
'liveness', 'valence']
#4 level: 1 2 3 4
mood_dict = dict()

for idx,name in enumerate(mood_name):
    lower_range = float(int(mood_value[idx]) - 1) / 4
    upper_range = float(int(mood_value[idx])) / 4
    value = random.uniform(lower_range,upper_range)
    mood_dict[name] = value

def euclide(x):
    distance = math.sqrt(
        np.sum(
            np.square([(x[n] - mood_dict[n]) for n in mood_name])
        )
    )
    return distance

#read data file
df = pd.read_csv("./primer_suggestion.csv")
#convert genres string to list
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x))

def genre_filter(x):
    if genre == "":
        return True
    is_contain = [genre in t for t in x['genres']]
    return any(is_contain)

#finding 
dist_df = df.apply(lambda x : pd.Series([x['artist'],x['song'],x['key'],x['time_signature'],x['genres'],euclide(x)],
index=['artist','song','key','time_signature','genres','dist']),axis=1)

dist_df = dist_df[dist_df.apply(lambda x: genre_filter(x),axis=1)]
sort_df = dist_df.sort_values('dist')

#get top 10
result = sort_df.head(15)

if list(result.index.values) == []:
    print("Check your genre please")
    exit()
rand_idx = random.choice(list(result.index.values))

print("Artist name: {}".format(result['artist'][rand_idx]))
print("Primer song: {}".format(result['song'][rand_idx]))
print("Suggested key: {}".format(result['key'][rand_idx]))
print("Suggested time signature: {}/4".format(result['time_signature'][rand_idx]))
print("Genre of primer song: {}".format(result['genres'][rand_idx]))
print("user's mood: {}".format(mood_dict))
print("Song closeness value: {}".format(result['dist'][rand_idx]))


