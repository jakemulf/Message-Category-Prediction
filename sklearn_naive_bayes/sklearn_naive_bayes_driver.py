"""
sklearn_naive_bayes_driver.py

Driver file for the sklearn naive bayes implimentation.  Takes user input of training data and testing data
along with the desired function to make predictions
"""

usage = """
python3 sklearn_naive_bayes_driver.py <test_data> <train_data> <function> <pre_filter_function> <post_filter_function>
"""

from sklearn.naive_bayes import MultinomialNB

import make_2d_array, functions, pre_filter_functions, post_filter_functions


FUNCTIONS = {
    'None': None,
    'get_synonym_sets': functions.get_synonym_sets,
}

PRE_FILTER_FUNCTIONS = {
    'None': None,
    'filter_by_similarity': pre_filter_functions.filter_by_similarity
}

POST_FILTER_FUNCTIONS = {
    'None': None,
    'filter_by_features': post_filter_functions.filter_by_features,
}


def get_func(func, d):
    if func in d:
        return d[func]
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


def main(test, train, func, pre_filter_func, post_filter_func):
    func = get_func(func, FUNCTIONS)
    pre_filter_func = get_func(pre_filter_func, PRE_FILTER_FUNCTIONS)
    post_filter_func = get_func(post_filter_func, POST_FILTER_FUNCTIONS)
    data = make_2d_array.driver([test,train], func, pre_filter_func, post_filter_func)
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
        pre_filter_func = sys.argv[4]
        post_filter_func = sys.argv[5]
    except:
        print("usage: " + usage)
        sys.exit(-1)
    
    main(test, train, func, pre_filter_func, post_filter_func)
