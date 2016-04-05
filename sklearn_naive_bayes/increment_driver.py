"""
increment_driver.py

Takes a range of thresholds and an increment value and runs
the prediction model for each threshold
"""
usage = """
python3 sklearn_naive_bayes/increment_driver.py <test_data> <train_data> <threshold_start> <threshold_end> <threshold_increment>
"""
import post_filter_functions


def main(test, train, threshold_start, threshold_end, threshold_increment):
    values = []
    counts = 
    while threshold_start <= threshold_end:
        threshold_start += threshold_increment

    print(values)


if __name__ == '__main__':
    import sys
    try:
        test = sys.argv[1]
        train = sys.argv[2]
        func = sys.argv[3]
        threshold_start = float(sys.argv[4])
        threshold_end = float(sys.argv[5])
        threshold_increment = float(sys.argv[6])

    except:
        print("usage: " + usage)
        sys.exit(-1)

    main(test, train, threshold_start, threshold_end, threshold_increment)

