"""
increment_driver.py

Takes a range of thresholds and an increment value and runs
the prediction model for each threshold
"""
usage = """
python3 sklearn_naive_bayes/increment_driver.py <test_data> <train_data> <threshold_start> <threshold_end> <threshold_increment>
"""
import post_filter_functions, make_2d_array, sklearn_naive_bayes_driver


def main(test, train, threshold_start, threshold_end, threshold_increment):
    values = []
    (completed_array, input_categories) = make_2d_array.driver([test, train], None, None, None, None)
    (counts, message_length) = post_filter_functions.make_counts(completed_array, input_categories)
    while threshold_start <= threshold_end:
        ignore_columns = post_filter_functions.make_ignore_columns(counts, threshold_start, message_length)
        new_array = post_filter_functions.remove_columns(completed_array, ignore_columns)
        curr_prediction = sklearn_naive_bayes_driver.make_prediction([new_array, input_categories])
        values.append(curr_prediction)
        threshold_start += threshold_increment

    print(values)


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

