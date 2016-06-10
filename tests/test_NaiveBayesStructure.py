"""
Unit tests for NaiveBayesStructure.py
"""
import unittest, random

import NaiveBayesStructure as NBS

def between(target, lower, higher):
    return target >= lower and target <= higher

def within(lst, dist):
    low = -1
    high = -1

    for value in lst:
        curr_len = len(value)
        if -1 in [low, high]:
            low = curr_len
            high = curr_len

        if curr_len < low:
            low = curr_len
        elif curr_len > high:
            high = curr_len

        if high-low > dist:
            return False

    return True


class TestNBS(unittest.TestCase):
    all_file = 'data/all.csv'

    @classmethod
    def setUpClass(cls):
        cls.all_struct = NBS.NaiveBayesStructure(cls.all_file)

    def test_get_training_testing(self):
        """
        Test to make sure the ratio is correct
        """
        d1 = self.all_struct.get_training_testing(.1)
        d2 = self.all_struct.get_training_testing(.2)
        d3 = self.all_struct.get_training_testing(.3)

        self.assertTrue(between(len(d1['test'])/(len(d1['train']) + len(d1['test'])), .09, .11))
        self.assertTrue(between(len(d2['test'])/(len(d2['train']) + len(d2['test'])), .19, .21))
        self.assertTrue(between(len(d3['test'])/(len(d3['train']) + len(d3['test'])), .29, .31))

    def test_get_cross_validation_chunks(self):
        """
        Test to make sure the number of chunks returned is correct
        and that they are all equal (with minor variation) in length
        """
        c1 = self.all_struct.get_cross_validation_chunks(3)
        c2 = self.all_struct.get_cross_validation_chunks(30)
        c3 = self.all_struct.get_cross_validation_chunks(100)

        self.assertTrue(len(c1) == 3)
        self.assertTrue(len(c2) == 30)
        self.assertTrue(len(c3) == 100)

        self.assertTrue(within(c1, 1))
        self.assertTrue(within(c2, 1))
        self.assertTrue(within(c3, 1))

    def test_column_thresholds(self):
        """
        Test to make sure the column thresholds are sorted by
        threshold and the column index is sorted
        """
        last_threshold = 0
        for i in range(len(self.all_struct.column_thresholds)):
            if self.all_struct.column_thresholds[i].column != i:
                self.fail('Expected column to be {} but found {}'.format(
                    i, self.all_struct.column_thresholds[i].column))

            if self.all_struct.column_thresholds[i].threshold >= last_threshold:
                last_threshold = self.all_struct.column_thresholds[i].threshold
            else:
                self.fail('Thresholds not in sorted order')
