"""
post_filter_functions.py

Holds functions that take the completed 2d array and modifies it
"""
from math import log
import numpy


def make_ignore_columns(counts, threshold, message_length):
    """
    Determines which columns to ignore
    """
    ignore_columns = []

    for i in range(message_length):
        count = counts[i]
        #Sum up the counts of all categories for easier processing later
        all_messages = [0,0]
        for key in count.keys():
            all_messages[0] += count[key][0]
            all_messages[1] += count[key][1]

        #For each category, compute the needed values and determine if it fits the threshold
        threshold_not_met = 0
        for category in count.keys():
            cat_arr = count[category]
            occurr_in_cat = cat_arr[1] + 1
            messages_in_cat = sum(cat_arr)
            occurr_out_cat = all_messages[1] - cat_arr[1] + 1
            messages_out_cat = (all_messages[1] - cat_arr[1]) + (all_messages[0] - cat_arr[0])

            if abs(log( (occurr_in_cat/messages_in_cat) / (occurr_out_cat/messages_out_cat))) < threshold:
                threshold_not_met += 1

        if threshold_not_met == len(count.keys()):
            ignore_columns.append(i)


    return ignore_columns


def make_counts(completed_array, input_categories):
    """
    Makes the count dict. Also returns the length of the messages
    """
    test_message = completed_array[0]
    train_message = completed_array[1]
    test_category = input_categories[0]
    train_category = input_categories[1]

    #Transpose messages to make column traversal more efficient
    messages = numpy.array(test_message + train_message).transpose()
    categories = test_category + train_category

    counts = []

    for i in range(len(messages)):
        count = {}

        #Generate the 0/1 counts for the current word for each category
        for j in range(len(categories)):
            curr_category = categories[j]
            if curr_category not in count.keys():
                count[curr_category] = [0,0]

            count[curr_category][messages[i][j]] += 1

        counts.append(count)

    return (counts, len(messages))


def remove_columns(completed_array, ignore_columns):
    """
    Removes the columns from the array
    """
    print("items removed: " + str(len(ignore_columns)))
    
    if len(ignore_columns) == 0:
        return completed_array

    ignore_columns.sort()
    new_completed_array = []
    for arr in completed_array:
        new_array = []
        for row in arr:
            count = 0
            appender = []
            for i in range(len(row)):
                if count < len(ignore_columns) and ignore_columns[count] == i:
                    count += 1 
                else:
                    appender.append(row[i])

            new_array.append(appender)
        new_completed_array.append(new_array)

    return new_completed_array


def filter_by_features(completed_array, input_categories, threshold):
    """
    Takes the completed array and removes all the columns that have a variation less
    than the given threshold
    """
    (counts, message_length) = make_counts(completed_array, input_categories)
    ignore_columns = make_ignore_columns(counts, threshold, message_length)
    new_array = remove_columns(completed_array, ignore_columns)
    
    return new_array
