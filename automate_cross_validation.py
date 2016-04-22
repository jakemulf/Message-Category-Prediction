"""
automate_cross_validation.py

Takes a single csv file, breaks it up into chunks and runs cross_validation_folder_reading
on it.

The chunks are created multiple times based on user input, and the user also supplies input
for cross_validation_folder_reading
"""

usage = """
usage: python3 automate_cross_validation <base_file> <times_to_run> <chunk_destination> <percent_per_chunk>
    <threshold_start> <threshold_end> <threshold_increment> <picture_location>

The base file MUST be in the same directory as automate_cross_validation.py
"""
import subprocess

import file_creation.chunks as chunks
import sklearn_naive_bayes.cross_validation_folder_reading as cvfr

def main(base_file, times_to_run, chunk_destination, percent_per_chunk, threshold_start,
         threshold_end, threshold_increment, picture_location):
    if chunk_destination[-1] != '/':
        chunk_destination += '/'
    if picture_location[-1] != '/':
        picture_location += '/'

    while True:
        curr_input = input("WARNING: {} DIRECTORY IS ABOUT TO BE DELETED. ".format(chunk_destination) +
            "TYPE exit TO EXIT, OR HIT ENTER TO CONTINUE: ")
        if curr_input == 'exit':
            exit(0)
        elif curr_input == '':
            break

    subprocess.call(['rm', '-rf', chunk_destination])

    for i in range(times_to_run):
        # Make chunks
        subprocess.call(['mkdir', chunk_destination])
        file_contents = chunks.get_contents(base_file)
        chunks.write_files(file_contents, percent_per_chunk, chunk_destination + base_file)
        # Run cross validation
        data = cvfr.get_data(chunk_destination, threshold_start, threshold_end, threshold_increment)
        cvfr.graph_data(data, threshold_start, threshold_increment, picture_location + str(i) + '.png')
        # Delete created files
        subprocess.call(['rm', '-rf', chunk_destination]) 


if __name__ == '__main__':
    import sys
    try:
        base_file = sys.argv[1]
        times_to_run = int(sys.argv[2])
        chunk_destination = sys.argv[3]
        percent_per_chunk = float(sys.argv[4])
        threshold_start = float(sys.argv[5])
        threshold_end = float(sys.argv[6])
        threshold_increment = float(sys.argv[7])
        picture_location = sys.argv[8]
    except:
        print(usage)
        sys.exit(-1)

    main(base_file, times_to_run, chunk_destination, percent_per_chunk,
         threshold_start, threshold_end, threshold_increment, picture_location)
