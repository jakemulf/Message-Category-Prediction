"""
get_inaccurate_prediciton.py

Tells us what messages were inaccurately predicted
"""

usage = """
usage: python3 get_inaccurate_prediction.py <test_data> <train_data> <threshold>
"""

from sklearn.naive_bayes import MultinomialNB
from numpy import mean

from sklearn_naive_bayes import make_2d_array, post_filter_functions, sklearn_naive_bayes_driver


def make_not_predicted(d1, d2):
    inaccurate_results = []
    total = min(len(d1), len(d2))
    for i in range(total):
        if d1[i] != d2[i]:
            inaccurate_results.append(i)

    return inaccurate_results


def get_not_predicted(data):
    gnb = MultinomialNB()
    not_predicted = gnb.fit(data[0][1], data[1][1]).predict(data[0][0])
    return make_not_predicted(not_predicted, data[1][0])


def get_confusion_matrix(actual, predicted):
    """
    Creates a dictionary structure x:y -> Z where x is the actual value
    and y is the predicted value and Z is the sum of that occurrence
    """
    confusion_dict = {}
    for i in range(min(len(predicted),len(actual))):
        curr_actual = actual[i]
        curr_predict = predicted[i]
        curr_key = (curr_actual, curr_predict)

        if curr_key in confusion_dict.keys():
            confusion_dict[curr_key] += 1
        else:
            confusion_dict[curr_key] = 1

    return confusion_dict


def average_message_length(messages, indexs, in_indexs):
    lengths = []

    if in_indexs:
        for i in indexs:
            lengths.append(len(messages[i]))

    else:
        curr_index = 0
        for i in range(len(messages)):
            if curr_index >= len(indexs) or indexs[curr_index] != i:
                lengths.append(len(messages[i]))
            else:
                curr_index += 1

    return mean(lengths)


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



def main(test_data, train_data, threshold):
    unique_words, messages, categories = make_2d_array.make_messages([test_data, train_data])
    arrays = make_2d_array.make_message_arrays(unique_words, messages, None, post_filter_functions.filter_by_features)
    data = [post_filter_functions.filter_by_features(arrays, categories, threshold), categories]

    gnb = MultinomialNB()
    predicted = gnb.fit(data[0][1], data[1][1]).predict(data[0][0])
    actual = data[1][0]

    print(_predict_results(predicted, actual))


if __name__ == '__main__':
    import sys
    try:
        test_data = sys.argv[1]
        train_data = sys.argv[2]
        threshold = float(sys.argv[3])
    except:
        print(usage)
        sys.exit(-1)

    main(test_data, train_data, threshold)
