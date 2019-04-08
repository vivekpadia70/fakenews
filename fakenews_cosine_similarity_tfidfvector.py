#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:15:27 2019

@author: vivek
"""

from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.keys import Keys
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

option = webdriver.ChromeOptions()
option.add_argument("--incognito")

browser = webdriver.Chrome(executable_path='./chromedriver', options=option)

browser.get('https://www.google.com/webhp?tbm=nws')

queryString = "Pakistan announces release of 360 Indian fishermen in jail"

browser.find_element_by_xpath("//input[@title='Search']").send_keys(queryString, Keys.ENTER)

headings = browser.find_elements_by_xpath("//div[@class='g']//h3")

sentences = []
sentences.append(queryString)
for heading in headings:
    sentences.append(heading.text)

print(sentences)

count_vectorizer = TfidfVectorizer(stop_words='english')
sparze_matrix = count_vectorizer.fit_transform(sentences)

doc_matrix = sparze_matrix.todense()
df = pd.DataFrame(doc_matrix)

print(cosine_similarity(df, df)[0])
