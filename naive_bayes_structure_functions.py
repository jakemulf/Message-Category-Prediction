"""
naive_bayes_structure_functions.py

Holds functions to compare instances of NaiveBayesStructure
classes and their data structures
"""
from sklearn.naive_bayes import MultinomialNB
from numpy import array

def compare_class(test, train):
    """
    Compares 2 NaiveBayesStructure contents
    """
    return compare_structure(test.contents, train.contents)


def compare_structure(test, train):
    """
    Compares training and testing data for accuracy
    of the model
    """
    np_test = array(test)
    np_train = array(train)

    return _compare_info(np_test, np_train)

def _compare_info(np_test, np_train):
    """
    Makes the MNB class to do the prediction
    """
    test_messages = np_test[:,0]
    test_category = np_test[:,1]

    train_messages = np_train[:,0]
    train_category = np_train[:,1]

    gnb = MultinomialNB()
    prediction = gnb.fit(list(train_messages), list(train_category)).predict(list(test_messages))

    return _predict_results(prediction, test_category)


def _predict_results(d1, d2):
    """
    Determines the ratio of accurate predictions
    """
    correct = 0
    total = min(len(d1),len(d2))
    for i in range(total):
        if d1[i] == d2[i]:
            correct += 1

    return correct/total
