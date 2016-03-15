"""
post_filter_functions.py

Holds functions that take the completed 2d array and modifies it
"""

def feature_threshold(completed_array, threshold):
    """
    Takes the completed array and removes all the columns that have a variation less
    than the given threshold
    
    change to this way: abs log (occurrences in category / messages in category)/(occurrences outside category / messages outside category)
    """
    ignore_columns = []
    for i in range(len(completed_array[0][0])):
        count = {}
        count[0] = 0
        count[1] = 0
        for arr in completed_array:
            for row in arr:
                count[row[i]] += 1

        total = count[0] + count[1]
        if count[1]/total <= threshold:
            ignore_columns.append(i)

    for arr in completed_array:
        for row in arr:
            count = 0
            for col in ignore_columns:
                row.pop(col-count) #remove element at index
                count += 1

    return completed_array


def filter_by_features(completed_array):
    """
    Takes the completed 2d array and filters out the values
    that are poor indicators of prediction
    """
    return feature_threshold(completed_array, .003)
