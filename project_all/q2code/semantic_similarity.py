import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import pandas as pd
import nltk


from pprint import pprint

def jaccard_similarity(string1, string2):
    list1 = nltk.word_tokenize(string1)
    list2 = nltk.word_tokenize(string2)
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

'''
list1 = ['dog', 'cat', 'cat', 'rat']
list2 = ['dog', 'cat', 'mouse']
print(jaccard_similarity(list1, list2))

'''

module_url = "../models/universal-sentence-encoder_4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)

def embed(input):
    return model(input)

def embedding_similarity(string1, string2):
    list_two_text = [string1, string2]
    embeddings = embed(list_two_text)
    sim = np.inner(embeddings[0], embeddings[1])
    return sim


df = pd.read_csv('../data/skeptics_answer_question_pair.csv')

list_of_feat_vectors = []

for idx, row in df.iterrows():
    answerid = row['answerid']
    answerbody = row['answerbody']
    questitle = row['questitle']
    quesbody = row['quesbody']
    title_jaccard = jaccard_similarity(questitle, answerbody)
    body_jaccard = jaccard_similarity(quesbody, answerbody)
    title_embed = embedding_similarity(questitle, answerbody)
    body_embed = embedding_similarity(quesbody, answerbody)
    print(title_jaccard, body_jaccard, title_embed, body_embed)
    feat_vec = {
        'id': answerid,
        'title_jaccard': title_jaccard,
        'body_jaccard': body_jaccard,
        'title_embed': title_embed,
        'body_embed': body_embed
    }
    list_of_feat_vectors.append(feat_vec)

df = pd.DataFrame(list_of_feat_vectors)
df.to_excel('../result/answer_semantic_similarity.xlsx', index=False, encoding='utf-8')
