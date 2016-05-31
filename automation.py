"""
automation.py

Contains functios for automating cross validation and
training/testing randomization
"""
import numpy, copy

import naive_bayes_structure_comparison as nbs_comparison


class PlotPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.x) + ',' + str(self.y)


class Threshold:
    def __init__(self, start, end, increment):
        self.start = start
        self.end = end
        self.increment = increment


def automate_randomization(naive_bayes_structure, percent_for_testing, times_to_run, threshold):
    """
    Runs randomized training/testing data on the give input
    """
    for i in range(times_to_run):
        print('run: ' + str(i))
        results = _randomize(naive_bayes_structure, percent_for_testing, threshold)
        print(results)


def _get_threshold_data(test_data, train_data, naive_bayes_structure, threshold):
    """
    Runs the comparison on the data at various thresholds
    """
    results = []
    curr_threshold = threshold.start
    while curr_threshold <= threshold.end:
        print("current threshold: " + str(curr_threshold))
        #Remove values for thresholds in training and testing
        curr_test_data = _remove_columns(test_data, naive_bayes_structure, curr_threshold)
        curr_train_data = _remove_columns(train_data, naive_bayes_structure, curr_threshold)
        
        curr_result = nbs_comparison.compare_structure(curr_test_data, curr_train_data)
        results.append(PlotPoint(curr_threshold,curr_result))

        curr_threshold += threshold.increment

    return results


def _randomize(naive_bayes_structure, percent_for_testing, threshold):
    """
    Randoimzed training and testing data on the give input
    """
    data_dict = naive_bayes_structure.get_training_testing(percent_for_testing)
    test_data = data_dict['test']
    train_data = data_dict['train']
    if threshold is None:
        return [PlotPoint(0,nbs_comparison.compare_structure(test_data, train_data))]
    else:
        return _get_threshold_data(test_data, train_data, naive_bayes_structure, threshold)


def _remove_columns(data, nbs, threshold):
    """
    Removes the columns from the data structure that don't meet the threshold
    """
    ignore_columns = _make_ignore_columns(nbs, threshold)
    new_data = []

    for content in data:
        row = copy.copy(content[0])
        for col in ignore_columns:
            row.pop(col)

        new_content = [row]
        new_content.extend(content[1:])
        new_data.append(new_content)
    
    return new_data


def _make_ignore_columns(nbs, threshold):
    """
    Removes the columns that don't meet the threshold
    """
    ignore_columns = []
    for col in nbs.column_thresholds:#reminder that nbs.column_thresholds is sorted by threshold
        if col.threshold > threshold:
            break
        ignore_columns.append(col.column)

    return sorted(ignore_columns, reverse=True)


def automate_cross_validation(naive_bayes_structure, chunks, times_to_run, threshold):
    """
    Runs cross validation multiple times on the given input
    """
    for i in range(times_to_run):
        print('run: ' + str(i))
        results = _cross_validation(naive_bayes_structure, chunks, threshold)
        print(results)


def _cross_validation(naive_bayes_structure, chunks, threshold):
    """
    Performs cross validation on the data input
    """
    results = []
    cross_validation_chunks = naive_bayes_structure.get_cross_validation_chunks(chunks)
    for i in range(chunks):
        print('chunk: ' + str(i))
        test_data = cross_validation_chunks[i]
        train_data = []
        for x in range(len(cross_validation_chunks)):
            if x != i:
                train_data.extend(cross_validation_chunks[x])
        
        if threshold is None:
            results.append([PlotPoint(0,nbs_comparison.compare_structure(test_data, train_data))])
        else:
            results.append(_get_threshold_data(test_data, train_data, naive_bayes_structure, threshold))

    return results
