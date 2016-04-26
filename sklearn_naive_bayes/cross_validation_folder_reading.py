"""
cross_validation_folder_reading.py

Reads a single folder holding 'chunks' of data and performs cross validation
on all the chunks by using each chunk for testing and using the remaining data
for training
"""

usage = """
usage: python3 cross_validation_folder_reading.py <folder_name> <threshold_start> <threshold_end> <threshold_increment> <picture_name>
"""
import os, subprocess, random
import matplotlib.pyplot as plt

from .increment_driver import increment_threshold


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

    return x_axis, threshold_increment


def graph_data(data, threshold_start, threshold_increment, picture_name):
    plt.clf()
    plt.ylim(-.1, 1.1)
    x_axis, threshold_increment = make_x_axis(data, threshold_start, threshold_increment)

    for values in data:
        plt.plot(x_axis, values, c=[random.random(), random.random(), random.random()])

    plt.xlabel('Threshold')
    plt.ylabel('Prediction Rate')
    plt.savefig(picture_name)


def get_data(folder, threshold_start, threshold_end, threshold_increment):
    """
    Creates the cross validation data
    """
    files = os.listdir(folder)
    train_file_name = 'train.csv'
    if folder[-1] != '/':
        folder += '/'

    all_prediction_data = []
    all_message_data = []

    for i in range(len(files)):
        print('on chunk: ' + str(i))
        test_file = files[i]
        open(folder+train_file_name, 'w')
        for j in range(len(files)):
            if j != i:
                subprocess.call('cat {} >> {}'.format(folder + files[j], 
                                                      folder + train_file_name),
                                shell=True)

        (prediction_data, message_data, _, _) = increment_threshold(folder + test_file, folder + train_file_name,
                            threshold_start, threshold_end, threshold_increment)
        all_prediction_data.append(prediction_data)
        all_message_data.append(message_data)
        subprocess.call(['rm', folder + train_file_name])

    return all_prediction_data, all_message_data


def main(folder, threshold_start, threshold_end, threshold_increment, picture_name):
    data = get_data(folder, threshold_start, threshold_end, threshold_increment)
    graph_data(data, threshold_start, threshold_increment, picture_name)

if __name__ == '__main__':
    import sys
    try:
        folder = sys.argv[1]
        threshold_start = float(sys.argv[2])
        threshold_end = float(sys.argv[3])
        threshold_increment = float(sys.argv[4])
        picture_name = sys.argv[5]
    except:
        print(usage)
        sys.exit(-1)

    main(folder, threshold_start, threshold_end, threshold_increment, picture_name)
