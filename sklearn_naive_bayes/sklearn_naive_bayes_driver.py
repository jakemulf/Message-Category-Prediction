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

from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
import make_2d_array
data = make_2d_array.driver(['data/test.csv', 'data/train.csv'])

#data[0][0]: test data message information
#data[0][1]: train data message information
#data[1][0]: test data category information
#data[1][1]: train data category information

prediction = gnb.fit(data[0][1], data[1][1]).predict(data[0][0])

#prediction: test data category prediction

print(compare_data(prediction,data[1][0]))
