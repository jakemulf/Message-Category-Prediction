"""
automation.py

Contains functios for automating cross validation and
training/testing randomization
"""
import numpy

import naive_bayes_structure_comparison as nbs_comparison


def automate_randomization(naive_bayes_structure, percent_for_testing, times_to_run):
    """
    Runs randomized training/testing data on the give input
    """
    for i in range(times_to_run):
        print('run: ' + str(i))
        results = _randomize(naive_bayes_structure, percent_for_testing)
        print(results)


def _randomize(naive_bayes_structure, percent_for_testing):
    """
    Randoimzed training and testing data on the give input
    """
    data_dict = naive_bayes_structure.get_training_testing(percent_for_testing)
    test_data = data_dict['test']
    train_data = data_dict['train']
    return nbs_comparison.compare_structure(test_data, train_data)


def automate_cross_validation(naive_bayes_structure, chunks, times_to_run):
    """
    Runs cross validation multiple times on the given input
    """
    for i in range(times_to_run):
        print('run: ' + str(i))
        results = _cross_validation(naive_bayes_structure, chunks)
        print(results)
        print(numpy.mean(results))


def _cross_validation(naive_bayes_structure, chunks):
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
