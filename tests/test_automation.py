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

    def test_automate_cross_validation(self):
        """
        Tests to make sure the format of automate_cross_validation is correct
        """
        no_thres_res = automation.automate_cross_validation(
                       self.all_struct, 2, 1, None)

        self.assertEqual(len(no_thres_res), 1) #Length equal to runs
        for res in no_thres_res:
            self.assertEqual(len(res), 2) #Length equal to chunks

        thres_res = automation.automate_cross_validation(
                    self.all_struct, 2, 1, automation.Threshold(0,1,.49))

        self.assertEqual(len(thres_res), 1) #Length equal to runs
        for res in thres_res:
            self.assertEqual(len(res), 2) #Length equal to chunks
            for r in res:
                self.assertEqual(len(r), 3) #Length equal to number of thresholds

    def test_automate_randomization(self):
        """
        Tests to make sure the format of automate_randomization is correct
        """
        no_thres_res = automation.automate_randomization(
                       self.all_struct, .1, 2, None)

        self.assertEqual(len(no_thres_res), 2) #Length equal to runs

        thres_res = automation.automate_randomization(
                    self.all_struct, .1, 1, automation.Threshold(0,1,.51))

        self.assertEqual(len(thres_res), 1) #Length equal to runs
        for res in thres_res:
            self.assertEqual(len(res), 2) #Length equal to number of thresholds

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
