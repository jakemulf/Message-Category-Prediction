"""
unit tests for naive_bayes_structure_comparison.py
"""
import unittest

import NaiveBayesStructure as NBS
import naive_bayes_structure_comparison as compare


DEFAULT_VALUE = 0.6740331491712708


class TestComparison(unittest.TestCase):
    all_file = 'data/all.csv'

    @classmethod
    def setUpClass(cls):
        cls.all_struct = NBS.NaiveBayesStructure(cls.all_file)

    def test_compare_structure_base(self):
        """
        The base comparison with the original testing and training data
        """
        test = self.all_struct.contents[0:543]
        train = self.all_struct.contents[543:]
        val = compare.compare_structure(test, train)

        self.assertEqual(val, DEFAULT_VALUE)
