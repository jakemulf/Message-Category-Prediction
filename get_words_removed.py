"""
get_words_removed.py

Figures out what words are removed at each threshold level
"""

usage = """
usage: python3 get_words_removed.py <data_file_1> <data_file_2> <output_file>
"""
import csv
from math import log

from sklearn_naive_bayes import make_2d_array, post_filter_functions


def make_threshold(counts, message_length):
    """
    Takes the counts dict and creates the threshold of removal for each word
    """
    counts_threshold = []

    for i in range(message_length):
        count = counts[i]
        #Sum up the counts of all categories for easier processing later
        all_messages = [0,0]
        for key in count.keys():
            all_messages[0] += count[key][0]
            all_messages[1] += count[key][1]

        #For each category, compute the needed values and determine if it fits the threshold
        best_threshold = 0
        for category in count.keys():
            cat_arr = count[category]
            occurr_in_cat = cat_arr[1] + 1
            messages_in_cat = sum(cat_arr)
            occurr_out_cat = all_messages[1] - cat_arr[1] + 1
            messages_out_cat = (all_messages[1] - cat_arr[1]) + (all_messages[0] - cat_arr[0])

            curr_threshold = abs(log( (occurr_in_cat/messages_in_cat) / (occurr_out_cat/messages_out_cat)))
            if curr_threshold > best_threshold:
                best_threshold = curr_threshold


        counts_threshold.append(best_threshold)

    return counts_threshold


def make_word_array(index_dict, count_thresholds):
    """
    Takes the index dictionary {word: index} and creates an array
    where arr[index] -> (word, threshold)

    Sorts by threshold
    """
    word_array = ['']*len(index_dict)

    for word in index_dict.keys():
        index = index_dict[word]
        word_array[index] = (word, count_thresholds[index])

    return sorted(word_array, key=lambda x: x[1])


def get_counts_and_index_dict(data_files):
    """
    Creates the count and index_dict structures for the datafiles
    """
    unique_words, messages, categories = make_2d_array.make_messages(data_files)
    index_dict = make_2d_array.make_index_dict(unique_words, None)
    arrays = make_2d_array.make_2d_arrays(index_dict, messages, None)
    counts, message_length = post_filter_functions.make_counts(arrays, categories)

    return counts, index_dict, message_length
   
   
def write_to_file(output_file, word_array):
    """
    Writes the output to a csv file
    """
    with open(output_file, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        for word_threshold in word_array:
            csv_writer.writerow(word_threshold)
 

def main(data_files, output_file):
    counts, index_dict, message_length = get_counts_and_index_dict(data_files)
    count_thresholds = make_threshold(counts, message_length)
    word_array = make_word_array(index_dict, count_thresholds)

    write_to_file(output_file, word_array)


if __name__ == "__main__":
    import sys
    try:
        data_files = [sys.argv[1], sys.argv[2]]
        output_file = sys.argv[3]
    except:
        print(usage)
        sys.exit(-1)

    main(data_files, output_file)
