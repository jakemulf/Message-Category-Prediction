"""
tests for automation.py
"""
import unittest

import automation
import NaiveBayesStructure as NBS

class TestAutomation(unittest.TestCase):
    all_file = 'data/all.csv'

    @classmethod
    def setUpClass(cls):
        cls.all_struct = NBS.NaiveBayesStructure(cls.all_file)

    def test_remove_columns(self):
        no_remove = automation._remove_columns(
            self.all_struct.contents, self.all_struct, 0)

        remove_some = automation._remove_columns(
            self.all_struct.contents, self.all_struct, 500)

        remove_all = automation._remove_columns(
            self.all_struct.contents, self.all_struct, len(self.all_struct.contents[0][0]))

        self.assertEqual(len(no_remove[0][0]), len(self.all_struct.contents[0][0]))
        self.assertEqual(len(remove_some[0][0]), len(self.all_struct.contents[0][0])-500)
        self.assertEqual(len(remove_all[0][0]), 0)
