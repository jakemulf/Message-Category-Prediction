"""
automation.py

Contains functios for automating cross validation and
training/testing randomization
"""
import numpy

import naive_bayes_structure_comparison as nbs_comparison


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


def _randomize(naive_bayes_structure, percent_for_testing, threshold):
    """
    Randoimzed training and testing data on the give input
    """
    data_dict = naive_bayes_structure.get_training_testing(percent_for_testing)
    test_data = data_dict['test']
    train_data = data_dict['train']
    if threshold is None:
        return [nbs_comparison.compare_structure(test_data, train_data)]

    results = []
    curr_threshold = threshold.start
    while curr_threshold <= threshold.end:
        print("current threshold: " + str(curr_threshold))
        #Remove values for thresholds in training and testing
        curr_test_data = _remove_columns(test_data, naive_bayes_structure, curr_threshold)
        curr_train_data = _remove_columns(train_data, naive_bayes_structure, curr_threshold)

        curr_result = nbs_comparison.compare_structure(curr_test_data, curr_train_data)
        results.append(curr_result)

        curr_threshold += threshold.increment

    return results


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
        results.append(nbs_comparison.compare_structure(test_data, train_data))

    return results
