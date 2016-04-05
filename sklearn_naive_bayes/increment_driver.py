"""
increment_driver.py

Takes a range of thresholds and an increment value and runs
sklearn_naive_bayes_driver.return_main for each threshold
"""
usage = """
python3 sklearn_naive_bayes/increment_driver.py <test_data> <train_data> <function> <pre_filter_function> <post_filter_function> <threshold_start> <threshold_end> <threshold_increment>
"""
import sklearn_naive_bayes_driver

def main(test, train, func, pre_filter_func, post_filter_func, threshold_start, threshold_end, threshold_increment):
    values = []
    while threshold_start <= threshold_end:
        values.append(sklearn_naive_bayes_driver.return_main(test, train, func, pre_filter_func, post_filter_func, threshold_start))
        threshold_start += threshold_increment

    print(values)


if __name__ == '__main__':
    import sys
    try:
        test = sys.argv[1]
        train = sys.argv[2]
        func = sys.argv[3]
        pre_filter_func = sys.argv[4]
        post_filter_func = sys.argv[5]
        threshold_start = float(sys.argv[6])
        threshold_end = float(sys.argv[7])
        threshold_increment = float(sys.argv[8])

    except:
        print("usage: " + usage)
        sys.exit(-1)

    main(test, train, func, pre_filter_func, post_filter_func, threshold_start, threshold_end, threshold_increment)

