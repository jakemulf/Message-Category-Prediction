"""
post_filter_functions.py

Holds functions that take the completed 2d array and modifies it
"""
from math import log


def feature_threshold(completed_array, input_categories, threshold):
    """
    Takes the completed array and removes all the columns that have a variation less
    than the given threshold
    
    change to this way: abs log (occurrences in category / messages in category)/(occurrences outside category / messages outside category)
    """
    test_message = completed_array[0]
    train_message = completed_array[1]
    test_category = input_categories[0]
    train_category = input_categories[1]

    messages = test_message + train_message
    categories = test_category + train_category
    ignore_columns = []

    print('all columns: ' + str(len(messages[0])))
    for i in range(len(messages[0])):
        if i > 300:
            break
        print('column: ' + str(i))
        count = {}

        for j in range(len(messages)):
            curr_category = categories[j]
            if curr_category not in count:
                count[curr_category] = [0,0]
            for row in messages:
                count[curr_category][row[j]] += 1

        all_messages = [0,0]
        for key in count.keys():
            all_messages[0] += count[key][0]
            all_messages[1] += count[key][1]

        for category in count.keys():
            cat_arr = count[category]
            occurr_in_cat = cat_arr[1]
            messages_in_cat = sum(cat_arr)
            occurr_out_cat = all_messages[1] - cat_arr[1]
            messages_out_cat = (all_messages[1] - cat_arr[1]) + (all_messages[0] - cat_arr[0])

            if abs(log( (occurr_in_cat/messages_in_cat) / (occurr_out_cat/messages_out_cat))) < threshold:
                ignore_columns.append(i)
                break

    print(len(ignore_columns))
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
    return feature_threshold(completed_array, input_categories, .5)
