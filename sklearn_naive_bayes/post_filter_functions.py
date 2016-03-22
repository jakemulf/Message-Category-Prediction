"""
post_filter_functions.py

Holds functions that take the completed 2d array and modifies it
"""
from math import log
import numpy


def feature_threshold(completed_array, input_categories, threshold):
    """
    Takes the completed array and removes all the columns that have a variation less
    than the given threshold
    """
    test_message = completed_array[0]
    train_message = completed_array[1]
    test_category = input_categories[0]
    train_category = input_categories[1]

    #Transpose messages to make column traversal more efficient
    messages = numpy.array(test_message + train_message).transpose()
    categories = test_category + train_category
    ignore_columns = []

    for i in range(len(messages)):
        count = {}

        #Generate the 0/1 counts for the current word for each category
        for j in range(len(categories)):
            curr_category = categories[j]
            if curr_category not in count.keys():
                count[curr_category] = [0,0]

            count[curr_category][messages[i][j]] += 1

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

    print("items removed: " + str(len(ignore_columns)))
    for arr in completed_array:
        for row in arr:
            count = 0
            for col in ignore_columns:
                row.pop(col-count) #remove element at index
                count += 1

    return completed_array


def filter_by_features(completed_array, input_categories):
    """
    Takes the completed 2d array and filters out the values
    that are poor indicators of prediction
    """
    return feature_threshold(completed_array, input_categories, .98)
