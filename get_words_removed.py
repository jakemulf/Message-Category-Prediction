"""
get_words_removed.py

Figures out what words are removed at each threshold level
"""

usage = """
usage: python3 get_words_removed.py <data_file_1> <data_file_2> <threshold_start>
    <threshold_end> <threshold_increment> <output_file>
"""
from math import log

from sklearn_naive_bayes import make_2d_array, post_filter_functions


MAX = 500


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
        lowest_threshold = MAX
        for category in count.keys():
            cat_arr = count[category]
            occurr_in_cat = cat_arr[1] + 1
            messages_in_cat = sum(cat_arr)
            occurr_out_cat = all_messages[1] - cat_arr[1] + 1
            messages_out_cat = (all_messages[1] - cat_arr[1]) + (all_messages[0] - cat_arr[0])

            curr_threshold = abs(log( (occurr_in_cat/messages_in_cat) / (occurr_out_cat/messages_out_cat)))
            if curr_threshold < lowest_threshold:
                lowest_threshold = curr_threshold


        counts_threshold.append(lowest_threshold)

    return counts_threshold


def make_word_array(index_dict, count_thresholds):
    """
    Takes the index dictionary {word: index} and creates an array
    where arr[index] -> (word, threshold)
    """
    word_array = ['']*len(index_dict)

    for word in index_dict.keys():
        index = index_dict[word]
        word_array[index] = (word, count_thresholds[index])

    return word_array
 

def main(data_files, threshold_start, threshold_end, threshold_increment, output_file):
    # Get counts, index_dict
    unique_words, messages, categories = make_2d_array.make_messages(data_files)
    index_dict = make_2d_array.make_index_dict(unique_words, None)
    arrays = make_2d_array.make_2d_arrays(index_dict, messages, None)

    counts, message_length = post_filter_functions.make_counts(arrays, categories)
    # Make threshold for each count
    count_thresholds = make_threshold(counts, message_length)
    # Turn counts to array (for immediate lookup)
    word_array = make_word_array(index_dict, count_thresholds)
    print(word_array)
    # Go through each threshold and print to file when that word will be removed


if __name__ == "__main__":
    import sys
    try:
        data_files = [sys.argv[1], sys.argv[2]]
        threshold_start = float(sys.argv[3])
        threshold_end = float(sys.argv[4])
        threshold_increment = float(sys.argv[5])
        output_file = sys.argv[6]
    except:
        print(usage)
        sys.exit(-1)

    main(data_files, threshold_start, threshold_end, threshold_increment, output_file)
