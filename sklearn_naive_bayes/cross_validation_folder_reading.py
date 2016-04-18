"""
cross_validation_folder_reading.py

Reads a single folder holding 'chunks' of data and performs cross validation
on all the chunks by using each chunk for testing and using the remaining data
for training
"""

usage = """
usage: python3 cross_validation_folder_reading.py <folder_name> <threshold_start> <threshold_end> <threshold_increment>
"""
import os, subprocess

from increment_driver import increment_threshold


def main(folder, threshold_start, threshold_end, threshold_increment):
    files = os.listdir(folder)
    train_file_name = 'train.csv'
    if folder[-1] != '/':
        folder += '/'

    for i in range(len(files)):
        test_file = files[i]
        open(folder+train_file_name, 'w')
        for j in range(len(files)):
            if j != i:
                subprocess.call('cat {} >> {}'.format(folder + files[j], 
                                                      folder + train_file_name),
                                shell=True)

        increment_threshold(folder + test_file, folder + train_file_name,
                            threshold_start, threshold_end, threshold_increment)
        subprocess.call(['rm', folder + train_file_name])


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
