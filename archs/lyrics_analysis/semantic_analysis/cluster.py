import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import math, sys
import numpy as np

def para_cosine_sim(group_sentences_1,group_sentences_2):
    """
    calculate similarity between 2 paragraph
    """
    text1 = '\n'.join(group_sentences_1)
    text2 = '\n'.join(group_sentences_2)

    stemmer = nltk.stem.porter.PorterStemmer()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    def stem_tokens(tokens):
        return [stemmer.stem(item) for item in tokens]

    def normalize(text):
        return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

    vectorizer = TfidfVectorizer(tokenizer=normalize,stop_words=None)

    def cosine_sim(text1,text2):
        tfidf = vectorizer.fit_transform([text1,text2])
        return ((tfidf * tfidf.T).A)[0,1]

    return cosine_sim(text1,text2)


def lyrics_cluster(lyrics,segment_boundary):
    """
    - Input: lurics and a list of segments boundary
    - Output: a dictionary:
        Key: Name of section
        Value: list of ssm
    """
    sentences = [t for t in lyrics.split("\n") if t != '']
    if segment_boundary is None:
        segment_boundary = []
    segment_boundary = [0] + segment_boundary + [len(sentences)]
    sections = [sentences[segment_boundary[i] : segment_boundary[i + 1]] for i in range(len(segment_boundary) - 1)]

    order = []
    for x in range(len(sections)):
        for y in range(x + 1,len(sections)):
            order.append( (x,y,para_cosine_sim(sections[x],sections[y])) )

    order = sorted(order,key=lambda x : x[2], reverse=True)

    section_ids = set(list(range(len(sections))))

    section_groups = collections.defaultdict(list)

    def section_exist(id):
        for key in section_groups:
            if id in section_groups[key]:
                return key
        return -1
    
    #order : list of tuple,
    #2 first element: ids of structure
    #last element: degree of similarity
    #Get connected ids
    if len(order) > 0:
        #softmax probability
        p = np.asarray([x for _,_,x in order])
        p = p / p.sum()
        order = [(i_1,i_2) for i_1,i_2,_ in order]
        max_key = -1
        while len(order) != 0:
            random_idx = np.random.choice(list(range(len(order))),p = p)
            id_1,id_2 = order[random_idx]
            del order[random_idx]
            p = np.delete(p,random_idx)
            p /= p.sum()
            key_1 = section_exist(id_1)
            key_2 = section_exist(id_2)
            if key_1 == -1 and key_2 == -1:
                section_groups[max_key + 1] = [id_1,id_2]
                max_key = max_key + 1
            elif key_1 == -1:
                section_groups[key_2].append(id_1)
            elif key_2 == -1:
                section_groups[key_1].append(id_2) 
    else:
        section_groups[0] = [0]

    for key in section_groups:
        section_groups[key] = sorted(section_groups[key])

    return sections, section_groups
        

