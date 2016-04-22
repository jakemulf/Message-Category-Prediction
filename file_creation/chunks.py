"""
chunks.py

Takes a single csv file and creates a new directory with
'chunks' of the file broken up based on the user input
"""

usage = """
usage = python3 <file_name> <percent_per_chunk>
"""

import csv, random


def get_contents(file_name):
    """
    Takes a csv file and creates a randomized list of its contents
    """
    file_contents = []

    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            file_contents.append(row)

    random.shuffle(file_contents)

    return file_contents


def make_contents(file_contents, number_of_files, rows_per_file, overflow_files):
    """
    Breaks up the file contents for file writing
    """
    contents = []
    start = 0
    end = rows_per_file

    for i in range(number_of_files):
        if i < overflow_files:
            end += 1

        contents.append(file_contents[start:end])

        start = end
        end += rows_per_file
        
    return contents


def write_files(file_contents, percent_per_chunk, file_name):
    """
    Writes the file contents to multiple files (chunks)
    """
    number_of_files = int(1/percent_per_chunk)
    rows_per_file = len(file_contents)//number_of_files # int division
    overflow_files = len(file_contents)%number_of_files

    contents = make_contents(file_contents, number_of_files, rows_per_file, overflow_files)

    for i in range(len(contents)):
        curr_file_name = file_name[:-4] + '_' + str(i) + '.csv'
        with open(curr_file_name, 'w+') as csv_file:
            csv_writer = csv.writer(csv_file)
            for content in contents[i]:
                csv_writer.writerow(content)


def main(file_name, percent_per_chunk):
    file_contents = get_contents(file_name)
    write_files(file_contents, percent_per_chunk, file_name)


if __name__ == '__main__':
    import sys
    try:
        file_name = sys.argv[1]
        percent_per_chunk = float(sys.argv[2])

    except:
        print(usage)
        sys.exit(-1)

    main(file_name, percent_per_chunk)
