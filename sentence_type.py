#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:53:15 2019

@author: vivek
"""

import nltk
import pickle

posts = nltk.corpus.nps_chat.xml_posts()[:10000]

def dialogue_act_features(post):
    features={}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

f = open('nltk_sentence_type_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

print(classifier.classify(dialogue_act_features('is trump going to hell for holiday')))