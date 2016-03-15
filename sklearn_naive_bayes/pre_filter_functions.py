"""
pre_filter_functions.py

Holds functions that take the unique words and all the messages and modifies
them accordingly
"""
import random
from nltk.corpus import wordnet as wn


SIMILARITY_THRESHOLD = .9


def get_synonym_sets(word):
    synonyms = wn.synsets(word)
    return synonyms


def check_similarity(word1, word2):
    syn1 = get_synonym_sets(word1)
    syn2 = get_synonym_sets(word2)
    """sim = False
    for s1 in syn1:
        for s2 in syn2:
            comparison = s1.wup_similarity(s2)
            if comparison is not None and comparison > SIMILARITY_THRESHOLD:
                sim = True
                break
        if sim:
            break
    return sim"""
    try:
        comparison = syn1[0].wup_similarity(syn2[0])
    except:
        return False

    return comparison is not None and comparison > SIMILARITY_THRESHOLD

def replace_messages(replacements, all_messages):
    """
    Replaces the words in messages to the base similar word
    """
    new_all_messages = []

    for messages in all_messages:
        new_messages = []
        for message in messages:
            new_message = []
            for word in message:
                add_word = True
            
                for key_word in replacements.keys():
                    if word in replacements[key_word]:
                        new_message.append(key_word)
                        add_word = False
                        break

                if add_word:
                    new_message.append(word)

            new_messages.append(new_message)

        new_all_messages.append(new_messages)        

    return new_all_messages


def replace_words(words):
    """
    Removes all the unneeded words from the original list and
    creates the replacement dictionary
    """
    word_count = len(words)
    print('all words: ' + str(word_count))
    curr_word = 0
    new_words = []
    replacements = {}

    for word in words:
        print(curr_word)
        if len(new_words) == 0:
            new_words.append(word)
        else:
            add_word = True
            for new_word in new_words:
                if check_similarity(new_word, word):
                    add_word = False 
                    similar_word = new_word
                    break

            if add_word:
                new_words.append(word)
            else:
                if similar_word not in replacements:
                    replacements[similar_word] = set()
                replacements[similar_word].add(word)

        curr_word += 1

    return new_words, replacements


def filter_by_similarity(words, messages):
    """
    Removes and replaces words that are 'too similar' to base words
    """
    words, replacements = replace_words(words)
    messages = replace_messages(replacements, messages)
    return words, messages
