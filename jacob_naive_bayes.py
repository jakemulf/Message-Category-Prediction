__author__ = 'Jacob Mulford'

import csv
import string
import sys
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

message_types = [
    'Action',
    'Community',
    'Information',
]

ignore_words = stopwords.words('english')

def remove_punctuation_and_lowercase(strn):
    """
    Removes the punctuation from the string
    and lowercases it.
    """
    char_list = []
    for char in strn:
        if not char in string.punctuation:
            char_list.append(char)
    word_list = ((''.join(char_list)).lower()).split(' ')

    return [word for word in word_list if word != '' and word not in ignore_words]
    #return [word for word in word_list if word != '']


def prior_probability(csv_contents):
    """
    Returns the prior probability of each
    message type occurring in the given data
    """
    message_counts = [0 for x in message_types]

    for i in range(1,len(csv_contents)):
        message_type = csv_contents[i][0]
        message_counts[message_types.index(message_type)] += 1

    for i in range(len(message_counts)):
        message_counts[i] += 1.0 # For Laplace Smoothing and casting to float
    count_total = sum(message_counts) + 3
    for i in range(len(message_counts)):
        message_counts[i] /= count_total

    return message_counts


def likelihood(csv_contents):
    """
    Returns the likelihood probability of each word
    in each category
    """
    word_counts = [{} for x in message_types]
    unique_words = set()

    for i in range(1,len(csv_contents)):
        index = message_types.index(csv_contents[i][0])
        for word in remove_punctuation_and_lowercase(csv_contents[i][1]):
            if word in word_counts[index]:
                word_counts[index][word] += 1
            else:
                word_counts[index][word] = 1
            unique_words.add(word)

    likelihood_dict = []
    unique_word_count = len(unique_words)

    for category_dictionary in word_counts:
        curr_likelihood = {}
        total_word_count = 0
        for word in category_dictionary:
            total_word_count += category_dictionary[word]

        for word in category_dictionary:
            curr_likelihood[word] = (category_dictionary[word]+1)/(unique_word_count + total_word_count)

        curr_likelihood['denominator count'] = unique_word_count + total_word_count
        likelihood_dict.append(curr_likelihood)

    return likelihood_dict


def message_counter(index, message, prior_probability_arr, likelihood_dict):
    probability = prior_probability_arr[index]

    for word in remove_punctuation_and_lowercase(message):
        if word in likelihood_dict[index]:
            probability *= likelihood_dict[index][word]
        else:
            probability /= likelihood_dict[index]['denominator count']

    if probability == 0:
        return sys.float_info.min
    return probability


def total_probability(message, prior_probability_arr, likelihood_dict):
    """
    Gives the total probability of a message occurring in the dataset
    """
    probability = 0.0

    for i in range(len(prior_probability_arr)):
        probability += message_counter(i, message, prior_probability_arr, likelihood_dict)

    return probability


def posterior_probability(message, category, prior_probability_arr, likelihood_dict):
    """
    Computes the probability that a message will appear in a specific category
    """
    message_index = message_types.index(category)

    numerator = message_counter(message_index, message, prior_probability_arr, likelihood_dict)
    denominator = total_probability(message, prior_probability_arr, likelihood_dict)

    return numerator/denominator

def read_csv_file(file_name):
    """
    Returns the contents of the csv file as an array
    """
    csv_contents = []
    with open(file_name) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            csv_contents.append(row)
    return csv_contents


def test_probability(test_file_name, train_file_name):
    """
    Attempts to map each message in the test file to the data
    from the train file
    """
    train_data = read_csv_file(train_file_name)
    prior_probability_arr = prior_probability(train_data)
    likelihood_dict = likelihood(train_data)

    test_data = read_csv_file(test_file_name)

    correct = 0

    for i in range(1,len(test_data)):
        best_type = message_types[0]
        for j in range(len(message_types)):
            test_type = message_types[j]
            curr_post = posterior_probability(test_data[i][1], best_type, prior_probability_arr, likelihood_dict)
            test_post = posterior_probability(test_data[i][1], test_type, prior_probability_arr, likelihood_dict)

            if test_post > curr_post:
                best_type = test_type
        if best_type == test_data[i][0]:
            correct += 1

    return correct*1.0/(len(test_data)-1)

def plot_data(test_file_name, train_file_name):
    """
    Plots the datapoints where the x value is P('Action'|Message) and
    the y value is P('Community'|Message)
    """
    train_data = read_csv_file(train_file_name)
    prior_probability_arr = prior_probability(train_data)
    likelihood_dict = likelihood(train_data)

    test_data = read_csv_file(test_file_name)

    x_values = [[] for x in message_types]
    y_values = [[] for x in message_types]

    for i in range(1, len(test_data)):
        x_values[message_types.index(test_data[i][0])].append(posterior_probability(test_data[i][1], 'Action', prior_probability_arr, likelihood_dict))
        y_values[message_types.index(test_data[i][0])].append(posterior_probability(test_data[i][1], 'Community', prior_probability_arr, likelihood_dict))

    message_points = [
        'ro',
        'go',
        'bo',
    ]
    for i in range(len(x_values)):
        plt.plot(x_values[i], y_values[i], message_points[i])
    plt.show()

print(test_probability('data/test.csv','data/train.csv'))
#plot_data('test.csv','train.csv')
