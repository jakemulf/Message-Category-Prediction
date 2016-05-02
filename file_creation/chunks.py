"""
chunks.py

Takes a single csv file and creates a new directory with
'chunks' of the file broken up based on the user input
"""

usage = """
usage = python3 <file_name> <percent_per_chunk>
"""

import csv, random


def make_categorized_contents(file_contents):
    """
    Takes the file contents and splits them up into their categories

    The dict will be structured like this:
        dict[category] -> [[messages], ratio]
    """
    categorized_contents = {}
    category_counts = {}
    
    for row in file_contents:
        category = row[0]
        message = row[1]

        if category in categorized_contents.keys():
            categorized_contents[category][0].append(message)
            category_counts[category] += 1

        else:
            categorized_contents[category] = [[message],0]
            category_counts[category] = 1

    message_count = len(file_contents)
    for category in categorized_contents.keys():
        category_count = category_counts[category]
        categorized_contents[category][1] = category_count/message_count

    return categorized_contents


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

    categorized_contents = make_categorized_contents(file_contents)

    return categorized_contents


def make_contents(categorized_contents, number_of_files, rows_per_file):
    """
    Breaks up the file contents for file writing
    """
    contents = []
    category_index = {cat:0 for cat in categorized_contents.keys()}
    category_increment = {}
    for cat in categorized_contents.keys():
        curr_ratio = categorized_contents[cat][1]
        curr_inc = rows_per_file*curr_ratio
        category_increment[cat] = int(curr_inc)


    for i in range(number_of_files):
        curr_indexs = {cat:(category_index[cat], category_index[cat] + category_increment[cat]) for cat in categorized_contents.keys()}

        for cat in category_index.keys():
            category_index[cat] = curr_indexs[cat][1]

        curr_contents = []
        for cat in categorized_contents.keys():
            (start, end) = curr_indexs[cat]
            for message in categorized_contents[cat][0][start:end]:
                curr_contents.append((cat, message))

        contents.append(curr_contents)

    overflow_index = 0
    for cat in category_index.keys():
        curr_index = category_index[cat]
        while curr_index < len(categorized_contents[cat][0]):
            message = categorized_contents[cat][0][curr_index]
            contents[overflow_index].append((cat, message))

            overflow_index += 1
            if overflow_index == number_of_files:
                overflow_index = 0
            curr_index += 1
        
    return contents


def write_files(categorized_contents, percent_per_chunk, file_name):
    """
    Writes the file contents to multiple files (chunks)
    """
    number_of_files = int(1/percent_per_chunk)
    number_of_rows = sum([len(categorized_contents[x][0]) for x in categorized_contents])
    rows_per_file = number_of_rows//number_of_files # int division

    contents = make_contents(categorized_contents, number_of_files, rows_per_file)

    for i in range(len(contents)):
        curr_file_name = file_name[:-4] + '_' + str(i) + '.csv'
        with open(curr_file_name, 'w+') as csv_file:
            csv_writer = csv.writer(csv_file)
            for content in contents[i]:
                csv_writer.writerow(content)


def main(file_name, percent_per_chunk):
    categorized_contents = get_contents(file_name)
    write_files(categorized_contents, percent_per_chunk, file_name)


if __name__ == '__main__':
    import sys
    try:
        file_name = sys.argv[1]
        percent_per_chunk = float(sys.argv[2])

    except:
        print(usage)
        sys.exit(-1)

    main(file_name, percent_per_chunk)
