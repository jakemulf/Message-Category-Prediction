"""
functions.py

Holds functions for word processing
"""
from nltk.corpus import wordnet as wn

#List generating functions
def get_synonym_sets(word):
    synonym_set = []
    synonyms = wn.synsets(word)
    for syn in synonyms:
        for lemma in syn.lemmas():
                synonym_set.append(lemma.name())
    return synonym_set

#Yes/No word filtering functions with a filter parameter
def get_important_words(word, important_words):
    if word in important_words:
        return word
    else:
        return None
