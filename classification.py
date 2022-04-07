import numpy as np
import pandas as pd
import regex as re
import nltk

dfMenuName = pd.read_csv('/Users/sohambose/Harvard/Sleek/kaggleMenuDataset/Menu.csv')
dfItemName = pd.read_csv('/Users/sohambose/Harvard/Sleek/kaggleMenuDataset/Dish.csv')
restaurantNames = dfMenuName['sponsor']
itemNames = dfItemName['name']

corpus = []
for row in itemNames:
    corpus.append(row)

#not sure if a good idea to lower case the words and replace non words with spaces
for i in range(len(corpus)):
    corpus[i] = corpus[i].lower()
    corpus[i] = re.sub(r'\W',' ',corpus[i])
    #corpus[i] = re.sub(r'\s+',' ',corpus[i])

wordFrequency = {}
for item in corpus:
    tokens = nltk.word_tokenize(item)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1