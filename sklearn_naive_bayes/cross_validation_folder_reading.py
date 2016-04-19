"""
cross_validation_folder_reading.py

Reads a single folder holding 'chunks' of data and performs cross validation
on all the chunks by using each chunk for testing and using the remaining data
for training
"""

usage = """
usage: python3 cross_validation_folder_reading.py <folder_name> <threshold_start> <threshold_end> <threshold_increment>
"""
import os, subprocess, random
import matplotlib.pyplot as plt

from increment_driver import increment_threshold


def get_max_or_min(data, func):
    """
    Gets the max or min value from the array of arrays
    """
    if func == max:
        find_max = True
    else:
        find_max = False

    if len(data) == 0:
        return None

    best_value = func(data[0])
    for arr in data:
        curr_value = func(arr)
        if (find_max and curr_value > best_value) or (not find_max and curr_value < best_value):
            best_value = curr_value

    return best_value


def make_x_axis(data, threshold_start, threshold_increment):
    """
    Makes the x axis for the data plotting
    """
    if len(data) == 0:
        return []

    x_axis = []
    
    for value in data[0]:
        x_axis.append(threshold_start)
        threshold_start += threshold_increment

    return x_axis


def graph_data(data, threshold_start, threshold_increment):
    x_axis = make_x_axis(data, threshold_start, threshold_increment)
    max_threshold = get_max_or_min(data, max)
    min_threshold = get_max_or_min(data, min)

    for values in data:
        plt.plot(x_axis, values, c=[random.random(), random.random(), random.random()])

    plt.show()

def main(folder, threshold_start, threshold_end, threshold_increment):
    files = os.listdir(folder)
    train_file_name = 'train.csv'
    if folder[-1] != '/':
        folder += '/'

    data = []

    for i in range(len(files)):
        test_file = files[i]
        open(folder+train_file_name, 'w')
        for j in range(len(files)):
            if j != i:
                subprocess.call('cat {} >> {}'.format(folder + files[j], 
                                                      folder + train_file_name),
                                shell=True)

        (curr_data, _, _) = increment_threshold(folder + test_file, folder + train_file_name,
                            threshold_start, threshold_end, threshold_increment)
        data.append(curr_data)
        subprocess.call(['rm', folder + train_file_name])

    graph_data(data, threshold_start, threshold_increment)

if __name__ == '__main__':
    import sys
    try:
        folder = sys.argv[1]
        threshold_start = float(sys.argv[2])
        threshold_end = float(sys.argv[3])
        threshold_increment = float(sys.argv[4])
    except:
        print(usage)
        sys.exit(-1)

    main(folder, threshold_start, threshold_end, threshold_increment)
