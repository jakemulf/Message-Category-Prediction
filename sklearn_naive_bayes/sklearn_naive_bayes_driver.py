"""
sklearn_naive_bayes_driver.py

Driver file for the sklearn naive bayes implimentation.  Takes user input of training data and testing data
along with the desired function to make predictions
"""

usage = """
python3 sklearn_naive_bayes_driver.py <test_data> <train_data> <function> <filter_function>
"""

from sklearn.naive_bayes import MultinomialNB

import make_2d_array, functions, filter_functions


FUNCTIONS = {
    'None': None,
    'get_synonym_sets': functions.get_synonym_sets
}

FILTER_FUNCTIONS = {
    'None': None,
    'filter_by_similarity': filter_functions.filter_by_similarity
}

def get_function(func):
    if func in FUNCTIONS:
        return FUNCTIONS[func]
    return None

def get_filter_function(func):
    if func in FILTER_FUNCTIONS:
        return FILTER_FUNCTIONS[func]
    return None

def compare_data(d1, d2):
    """
    Compares the 2 data arrays for accuracy
    """
    correct = 0
    total = min(len(d1),len(d2))
    for i in range(total):
        if d1[i] == d2[i]:
            correct += 1

    return correct/total


def main(test, train, func, filter_func):
    func = get_function(func)
    filter_func = get_filter_function(filter_func)
    data = make_2d_array.driver([test,train], func, filter_func)
    gnb = MultinomialNB()
    prediction = gnb.fit(data[0][1], data[1][1]).predict(data[0][0])

    #prediction: test data category prediction

    print(compare_data(prediction,data[1][0]))
    #data[0][0]: test data message information
    #data[0][1]: train data message information
    #data[1][0]: test data category information
    #data[1][1]: train data category information


if __name__ == '__main__':
    import sys
    try:
        test = sys.argv[1]
        train = sys.argv[2]
        func = sys.argv[3]
        filter_func = sys.argv[4]
    except:
        print("usage: " + usage)
        sys.exit(-1)
    
    main(test, train, func, filter_func)
