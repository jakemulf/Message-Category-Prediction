"""
automation.py

Contains functios for automating cross validation and
training/testing randomization
"""
import numpy, copy, random, os
from multiprocessing import Process, Queue, cpu_count
import matplotlib.pyplot as plt

import naive_bayes_structure_comparison as nbs_comparison

cpus = cpu_count()
PROCESS_LIMIT = cpus if cpus is not None else 1

PICTURE_DESTINATION = 'pictures/'


class PlotPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'


class Threshold:
    def __init__(self, start, end, increment):
        self.start = start
        self.end = end
        self.increment = increment


_random_color = lambda: [random.random(), random.random(), random.random()]
def graph_results(results, file_name):
    """
    Creates a plot graph for the results
    """
    plt.clf()
    plt.ylim(-.1,1.1)
    for result in results:
        result.sort(key=lambda a: a.x)
        color = _random_color()
        plt.plot([point.x for point in result], [point.y for point in result], c=_random_color())

    plt.xlabel('Threshold')
    plt.ylabel('Prediction Rate') 

    if not os.path.isdir(PICTURE_DESTINATION):
        os.mkdir(PICTURE_DESTINATION)

    plt.savefig(PICTURE_DESTINATION + file_name)


def process_results_top(results, func, plot_point):
    """
    Applies the function to the results structure from automate_cross_validation
    """
    for result in results:
        process_results(result, func, plot_point)


def process_results(results, func, plot_point):
    """
    Applies the function to the results structure from automate_randomization,
    or the inner structure of automate_cross_validation
    """
    for result in results:
        prediction, test = result[1]
        val = func(prediction, test)
        if plot_point:
            if type(val) is list:
                for v in val:
                    result.append(PlotPoint(result[0], v))
            else:
                result.append(PlotPoint(result[0], val))
        else:
            result.append(val)


###Functions for randomization###
def automate_randomization(nbs, percent_for_testing, times_to_run, threshold):
    """
    Runs randomized training/testing data on the give input
    """
    results = []
    for i in range(times_to_run):
        print('run: ' + str(i))
        results.append(_randomize(nbs, percent_for_testing, threshold))

    return results


def _randomize(nbs, percent_for_testing, threshold):
    """
    Randoimzed training and testing data on the give input
    """
    data_dict = nbs.get_training_testing(percent_for_testing)
    test_data = data_dict['test']
    train_data = data_dict['train']
    if threshold is None:
        prediction, test = nbs_comparison.compare_structure(test_data, train_data)
        return [(0, (prediction, test, test_data))]
    else:
        return _get_threshold_data(test_data, train_data, nbs, threshold)


###Functions for cross validation###
def automate_cross_validation(nbs, chunks, times_to_run, threshold):
    """
    Runs cross validation multiple times on the given input
    """
    results = []
    for i in range(times_to_run):
        print('run: ' + str(i))
        results.append(_cross_validation(nbs, chunks, threshold))

    return results


def _cross_validation(nbs, chunks, threshold):
    """
    Performs cross validation on the data input
    """
    results = []
    cross_validation_chunks = nbs.get_cross_validation_chunks(chunks)
    for i in range(chunks):
        print('chunk: ' + str(i))
        test_data = cross_validation_chunks[i]
        train_data = []
        for x in range(len(cross_validation_chunks)):
            if x != i:
                train_data.extend(cross_validation_chunks[x])
        
        if threshold is None:
            prediction, test = nbs_comparison.compare_structure(test_data, train_data)
            results.append([(0, (prediction, test, test_data))])
        else:
            results.append(_get_threshold_data(test_data, train_data, nbs, threshold))

    return results


###Shared functions###
def _get_threshold_data(test_data, train_data, nbs, threshold):
    """
    Runs the comparison on the data at various thresholds
    """
    #results = Queue()
    results = []
    curr_threshold = threshold.start
    processes = []
    while curr_threshold <= threshold.end:
        """
        #Make new process for _eval_threshold
        p = Process(target=_eval_threshold, args=(
                    test_data, train_data, nbs, curr_threshold, results,))
        processes.append(p)
        curr_threshold += threshold.increment
        """
        _eval_threshold(test_data, train_data, nbs, curr_threshold, results)
        curr_threshold += threshold.increment
    """
    i = 0
    while i < len(processes):
        print('process ' + str(i))
        start = i
        end = i+PROCESS_LIMIT
        for j in range(start, min(end, len(processes))):
            processes[j].start()
        for j in range(start, min(end, len(processes))):
            processes[j].join()

        i += PROCESS_LIMIT

    return [results.get() for p in processes]
    """
    return results


def _eval_threshold(test_data, train_data, nbs, curr_threshold, results):
    """
    Evaluates the comparison of the data at the given threshold
    """
    start_index = _make_start_index(nbs, curr_threshold)
    curr_test_data = _remove_columns(test_data, nbs, start_index)
    curr_train_data = _remove_columns(train_data, nbs, start_index)
    prediction, test = nbs_comparison.compare_structure(curr_test_data, curr_train_data)
    curr_word_info = _get_word_info(curr_test_data)
    curr_result = (prediction, test, curr_word_info)
    #results.put((curr_threshold,curr_result))
    results.append((curr_threshold,curr_result))


def _remove_columns(data, nbs, start_index):
    """
    Removes the columns from the data structure that don't meet the threshold
    """
    new_data = []

    for content in data:
        new_content = [content[0][start_index:]]
        new_content.extend(content[1:])
        new_data.append(new_content)
    
    return new_data


def _make_start_index(nbs, threshold):
    """
    Returns the starting column to slice
    """
    start_index = 0
    for col in nbs.column_thresholds:#reminder that nbs.column_thresholds is sorted by threshold
        if col.threshold > threshold:
            start_index = col.column
            break
    
    return start_index


def _get_word_info(data):
    """
    Returns the word info in order from the data
    """
    words = []
    for val in data:
        words.append(val[2])

    return words
