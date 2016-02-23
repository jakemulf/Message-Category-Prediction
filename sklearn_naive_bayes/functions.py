from nltk.corpus import wordnet as wn

def get_synonym_sets(word):
    synonym_set = []
    synonyms = wn.synsets(word)
    for syn in synonyms:
        for lemma in syn.lemmas():
                synonym_set.append(lemma.name())
    return synonym_set
