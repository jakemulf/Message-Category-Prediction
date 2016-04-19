"""
randomize.py

Takes 1 csv file and a % threshold (for testing) and creates
2 csv files, 1 training and 1 testing.

Default is without replacement, add --r for replacement
"""

usage = """
python3 randomize.py <file_name> <percent_for_testing>

options:
--r: replacement
"""
import csv
import random


def make_file_contents(file_name, percent_for_testing, replacement):
    """
    Takes a single csv file and creates a training and testing
    dataset randomly from that file.
    """
    file_contents = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            file_contents.append(row)

    all_indexs = list(range(len(file_contents)))
    break_index = int(len(file_contents)*percent_for_testing)

    if replacement:
        test_indexs = []
        for i in range(break_index):
            test_indexs.append(random.randint(0,len(all_indexs)-1))
        
        train_indexs = []
        for i in range(len(all_indexs)-break_index):
            train_indexs.append(random.randint(0,len(all_indexs)-1))

    else:
        random.shuffle(all_indexs)
        test_indexs = all_indexs[:break_index]
        train_indexs = all_indexs[break_index:]

    test_contents = [file_contents[x] for x in test_indexs]
    train_contents = [file_contents[x] for x in train_indexs]

    return test_contents, train_contents


def write_files(test_contents, train_contents, file_name):
    """
    Writes all the file contents to files
    """
    with open(file_name[:-4]+'_test.csv', 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        for content in test_contents:
            csv_writer.writerow(content)

    with open(file_name[:-4]+'_train.csv', 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        for content in train_contents:
            csv_writer.writerow(content)


def main(file_name, percent_for_testing, replacement):
    test_contents, train_contents = make_file_contents(file_name, percent_for_testing, replacement)
    write_files(test_contents, train_contents, file_name)


if __name__ == '__main__':
    import sys
    try:
        file_name = sys.argv[1]
        percent_for_testing = float(sys.argv[2])
    except:
        print(usage)
        sys.exit(-1)

    if len(sys.argv) > 3:
        if sys.argv[-1] == '--r':
            replacement = True
        else:
            print(usage)
            sys.exit(-1)

    else:
        replacement = False

    main(file_name, percent_for_testing, replacement)
