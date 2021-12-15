import spacy
from spacy.parts_of_speech import PROPN, ADJ, NUM, PRON, VERB
from textblob import TextBlob
import textstat
from nltk import word_tokenize, pos_tag
import re
from pprint import pprint
import pandas as pd
import os
import time
import datetime
import csv

import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 3000  # Set Duration To 1000 ms == 1 second

nlp = spacy.load ('en_core_web_lg')



def tense_calc(text):
    nltk_obj = word_tokenize (text)
    tagged = pos_tag (nltk_obj)

    tense = {}
    tense["future"] = len ([word for word in tagged if word[1] == ["MD", "VBC", "VBF"]])
    tense["present"] = len ([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]])
    tense["past"] = len ([word for word in tagged if word[1] in ["VBD", "VBN"]])
    return tense

def sentence_type_calc(text):
    doc = nlp (text)
    sentence_types = {'aff_sents': 0, 'neg_sents': 0, 'int_sents': 0, 'imp_sents': 0, 'opt_sents': 0, 'exc_sents': 0}
    for sent in doc.sents:
        # print(sent)
        # if sent.text.strip()
        clean_sent = sent.text.replace ('\"', '').strip ()
        if len (clean_sent) == 0:
            continue
        if clean_sent[-1] == '?':
            sentence_types['int_sents'] += 1
        elif clean_sent[-1] == '!':
            sentence_types['exc_sents'] += 1
        else:
            for token in sent:
                if token.is_punct == True:
                    continue
                if token.pos == VERB:
                    sentence_types['imp_sents'] += 1
                    break
                elif token.text.lower () in ['may', 'wish']:
                    sentence_types['opt_sents'] += 1
                    break
                elif token.dep_ == 'neg':
                    sentence_types['neg_sents'] += 1
                    break
                else:
                    sentence_types['aff_sents'] += 1
    return sentence_types

def pos_counts_calc(text):
    doc = nlp (text)
    pos_counts = {'proper_noun': 0, 'adjective': 0, 'cardinal_number': 0, 'pronoun': 0}
    for token in doc:
        if token.doc.is_tagged:
            if token.pos == PROPN:
                pos_counts['proper_noun'] += 1
            if token.pos == ADJ:
                pos_counts['adjective'] += 1
            if token.pos == NUM:
                pos_counts['cardinal_number'] += 1
            if token.pos == PRON:
                pos_counts['pronoun'] += 1
    return pos_counts

def feature_calc(filename):
    list_of_feature_vectors = []
    xcount = 0
    with open('../result/splits/done.txt', 'r', encoding='utf-8') as f:
        done = f.read ().splitlines ()
    with open('../result/splits/stubborn.txt', 'r', encoding='utf-8') as f:
        stubborn = f.read ().splitlines ()
    with open('../result/splits/error.txt', 'r', encoding='utf-8') as f:
        error = f.read ().splitlines ()
    fout = open('../result/splits/done.txt', 'a', encoding='utf-8')
    # sout = open('../result/splits/stubborn.txt', 'a')
    eout = open('../result/splits/error.txt', 'a', encoding='utf-8')
    with open (filename, 'r', encoding='utf-8') as fin:
        tsvreader = csv.reader(fin, dialect='excel')
        for idx, line in enumerate (tsvreader):
            # print(xcount)
            try:
                # print ('print line: ', line)
                # line = line.strip ()
                # print('h')
                id, text = line[0], ' '.join(line[1:])
                if not id.isdigit():
                    eout.write(id+'\n')
                    continue
                # if len(id)!=6:
                #     eout.write(id+'\n')
                #     continue
                if id in stubborn:
                    continue
                if id in done:
                    print(id, ' already done')
                    continue
                if id in error:
                    print('already marked error:\n', line)
                    continue
                print (xcount, idx, id)
                # print (text)
                '''tense'''
                tense = tense_calc(text)
                # print ('tense')
                '''sentence type'''
                sentence_types = sentence_type_calc(text)
                # print('sentence type')
                    # print(clean_sent, )
                '''readability'''
                readability = textstat.flesch_kincaid_grade (text)
                # print('readability')
                '''count'''
                doc = nlp (text)
                sentence_count = len (list (doc.sents))
                try:
                    word_count = 1.0 * len ([token.text for token in doc
                                         if token.is_stop != True and
                                         token.is_punct != True]) / sentence_count
                except:
                    word_count = 0
                # print('count')
                '''pos data'''
                pos_counts = pos_counts_calc(text)
                # print('pos data')
                '''sentiment and subjectivity'''
                sentiment_object = TextBlob(text)
                subjectivity = sentiment_object.subjectivity
                sentiment = sentiment_object.polarity
                # print('sentiment')
                '''presence of urls and images'''
                # print('x')
                # pat = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
                pat = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                urls = re.findall(pat, text)
                # print ('url')
                for u in range(len(urls)):
                    urls[u] = urls[u][0]
                url_count = len(urls)


                '''prepare feature vector'''
                feature_vector = {
                    'id': id,
                    'pronoun': pos_counts['pronoun'],
                    'present': tense['present'], 'past': tense['past'], 'future': tense['future'],
                    'sentiment': sentiment,
                    'affermative': sentence_types['aff_sents'], 'negative': sentence_types['neg_sents'],
                    'interrogative': sentence_types['int_sents'], 'imperative': sentence_types['imp_sents'],
                    'optative': sentence_types['opt_sents'], 'exclamation': sentence_types['exc_sents'],
                    'readability': readability,
                    'sentence_count': sentence_count, 'words_count_per_sentence': word_count,
                    'proper_noun': pos_counts['proper_noun'], 'adjective': pos_counts['adjective'], 'cardinal_number': pos_counts['cardinal_number'],
                    'subjectivity': subjectivity,
                    'url_img_count': url_count,
                }
                if 'title' in filename:
                    title_word_count = len ([token.text for token in doc
                                         if token.is_stop != True and
                                         token.is_punct != True])
                    feature_vector['title_words_count'] = title_word_count

                # pprint(feature_vector)
                # print()
                list_of_feature_vectors.append(feature_vector)
                xcount += 1
                fout.write(id+'\n')
                if xcount % 1500000 == 0:
                    break
            except Exception as e:
                try:
                    eout.write(line.strip()+'\n')
                except:
                    pass
                print(e)
            # time_elapsed = datetime.datetime.now () - time_start
            # print ('Elapsed:', time_elapsed)
            # if time_elapsed > datetime.timedelta(seconds=300):
            #     winsound.Beep (frequency, duration)
    eout.close()
    df = pd.DataFrame (list_of_feature_vectors)
    fout.close ()
    filepath = '../result/splits/features.' + '.'.join (
        filename.split ('_')[-1].split ('.')[:-1]) + '.xlsx'
    fcount = 1
    while os.path.exists (filepath):
        filepath = '../result/splits/features.' + '.'.join (
            filename.split ('_')[-1].split ('.')[:-1]) + str (fcount) + '.xlsx'
        fcount += 1
        pass
    df.to_excel (filepath, encoding='utf-8', index=False)
    fout.close ()
    winsound.Beep (frequency, duration)
    return


# feature_calc ('../data/clean_skeptics.questions.title.csv')
# feature_calc ('../data/clean_skeptics.questions.body.csv')
feature_calc ('../data/clean_skeptics.answers.body.csv')
# feature_calc ('../data/clean_cmv.questions.title.csv')
# feature_calc ('../data/clean_cmv.questions.body.csv')
# feature_calc ('../data/clean_cmv.answers.body.tsv')
