"""
increment_driver.py

Takes a range of thresholds and an increment value and runs
the prediction model for each threshold
"""
usage = """
python3 sklearn_naive_bayes/increment_driver.py <test_data> <train_data> <threshold_start> <threshold_end> <threshold_increment>
"""
import matplotlib.pyplot as plt

import post_filter_functions, make_2d_array, sklearn_naive_bayes_driver


def graph_values(values, threshold_start, threshold_increment):
    """
    Graphs the values generated with corresponding thresholds
    """
    x_axis = []
    max_value = max(values)
    min_value = min(values)
    min_threshold = threshold_start
    for value in values:
        x_axis.append(threshold_start)
        threshold_start += threshold_increment

    plt.plot(x_axis, values, 'ro')
    plt.axis([min_threshold - .05, (threshold_start - threshold_increment) + .05,
        min_value - .05, max_value + .05])
    plt.show()


def increment_threshold(test, train, threshold_start, threshold_end, threshold_increment):
    values = []
    (completed_array, input_categories) = make_2d_array.driver([test, train], None, None, None, None)
    (counts, message_length) = post_filter_functions.make_counts(completed_array, input_categories)
    old_start = threshold_start
    while threshold_start <= threshold_end:
        print('current threshold: ' + str(threshold_start))
        ignore_columns = post_filter_functions.make_ignore_columns(counts, threshold_start, message_length)
        new_array = post_filter_functions.remove_columns(completed_array, ignore_columns)
        curr_prediction = sklearn_naive_bayes_driver.make_prediction([new_array, input_categories])
        values.append(curr_prediction)
        threshold_start += threshold_increment

    graph_values(values, old_start, threshold_increment)


def main(test, train, treshold_start, threshold_end, threshold_increment):
    increment_threshold(test, train, threshold_start, threshold_end, threshold_increment)

if __name__ == '__main__':
    import sys
    try:
        test = sys.argv[1]
        train = sys.argv[2]
        threshold_start = float(sys.argv[3])
        threshold_end = float(sys.argv[4])
        threshold_increment = float(sys.argv[5])

    except:
        print("usage: " + usage)
        sys.exit(-1)

    main(test, train, threshold_start, threshold_end, threshold_increment)
