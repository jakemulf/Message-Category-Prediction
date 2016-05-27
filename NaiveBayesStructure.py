"""
NaiveBayesSctructure.py

Holds the class that contains the information for the naive bayes classifier
"""
import csv, string, random
from nltk.corpus import stopwords

IGNORE_WORDS = stopwords.words('english')


def make_file_contents(csv_file):
    """
    Creates the 2d array for the messages
    """
    csv_contents = _read_csv_file(csv_file)
    unique_words = _get_unique_words(csv_contents)
    index_dict = _get_index_dict(unique_words)
    file_contents = _transform_csv_contents(csv_contents, index_dict, len(unique_words))

    return file_contents


def _transform_csv_contents(csv_contents, index_dict, unique_words_count):
    """
    Creates the 0/1 array structure of words, and numbers the seen categories
    """
    file_contents = []
    seen_categories = [] # Category count should remain small enough for
        # array traversal to make minimal impact in performance
    for content in csv_contents:
        category = content.category
        words = content.words

        if not category in seen_categories:
            seen_categories.append(category)
        category_index = seen_categories.index(category)
        word_structure = [0]*unique_words_count

        for word in words:
            word_index = index_dict[word]
            word_structure[word_index] = 1

        file_contents.append([word_structure, category_index])

    return file_contents


def _get_index_dict(unique_words):
    """
    Makes the index dict for the unique words
    """
    index_dict = {}
    for i in range(len(unique_words)):
        index_dict[unique_words[i]] = i

    return index_dict


def _get_unique_words(csv_contents):
    """
    Gets an ordered list of all the unique words
    """
    unique_words = set()
    for content in csv_contents:
        words = content.words
        for word in words:
            unique_words.add(word)

    return list(unique_words)


def _remove_punctuation_and_lowercase(strn):
    """
    Removes the punctuation from the string
    and lowercases it.
    """
    char_list = []
    backslash_found = False
    for char in strn:
        if backslash_found:
            backslash_found = False
            if char == 'n':
                char_list.append(' ')
        elif not char in string.punctuation:
            char_list.append(char)
        elif char == '\\':
            backslash_found = True
    word_list = ((''.join(char_list)).lower()).split(' ')

    return [word for word in word_list if word != '' and word not in IGNORE_WORDS]


def _read_csv_file(file_name):
    """
    Returns the contents of the csv file as an array
    """
    csv_contents = []
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            category = row[0]
            words = _remove_punctuation_and_lowercase(row[1])
            csv_contents.append(CSVRow(category, words))
    return csv_contents


class CSVRow:
    def __init__(self, category, words):
        self.category = category
        self.words = words


class NaiveBayesStructure:
    def __init__(self, csv_file):
        self.contents = make_file_contents(csv_file)

    def get_training_testing(self, percent_for_testing):
        """
        Returns a randomized set of training and testing data based
        on the percent given
        """
        random.shuffle(self.contents)
        cutoff_index = int(len(self.contents)*percent_for_testing)
        dic = {}
        dic['test'] = self.contents[0:cutoff_index]
        dic['train'] = self.contents[cutoff_index:]

        return dic

    def get_cross_validation_chunks(self, number_of_chunks):
        """
        Breaks up the data into N chunks
        """
        random.shuffle(self.contents)
        count_per_chunk = len(self.contents)//number_of_chunks
        overflow_count = len(self.contents)%number_of_chunks

        cross_validation_chunks = []
        start = 0
        end = count_per_chunk
        for i in range(number_of_chunks):
            if i < overflow_count:
                end += 1

            cross_validation_chunks.append(self.contents[start:end])
            start = end
            end += count_per_chunk

        return cross_validation_chunks
