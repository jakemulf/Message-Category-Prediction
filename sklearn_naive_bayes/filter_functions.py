"""
filter_functions.py

Holds functions for filtering words
"""

SIMILARITY_THRESHOLD = .5

def filter_by_similarity(words):
    """
    Removes words in the list that are 'too similar' to
    other words alread in the list
    """
    new_words = []

    for word in words:
        if len(new_words) == 0:
            new_words.append(word)
        else:
            add_word = True
            for new_word in new_words:
                if """word too similar to new_word""": #TODO
                    add_word = False 
                    break

            if add_word:
                new_words.append(word)

    return new_words
