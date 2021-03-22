import unittest
from Projects.Assignment3.NeuralNet_Agent.agent import *


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()

    def test_deconstructSentence(self):
        test_String = "What hours are you open?"
        self.assertEqual(self.agent.deconstructSentence(test_String),
                         ['what', 'hour', 'are', 'you', 'open', '?'], "Should be ['what', 'hour', 'are', 'you', "
                                                                      "'open', '?']")

    def test_predictResponse(self):
        test_String = "What hours are you open?"
        self.assertEqual(self.agent.predictResponse(test_String),
                         [{'intent': 'hours', 'probability': '1.0'}], "Should be [{'intent': 'hours', 'probability': "
                                                                      "'1.0'}]")

    def test_getResponse(self):
        test_String = "What hours are you open?"
        predictedResponse = self.agent.predictResponse(test_String)
        self.assertTrue(self.agent.getResponse(predictedResponse) in ["Our hours are 10am-6pm every day",
                                                                      "We're open every day 10am-6pm"],
                        "Should be \"Our hours are 10am-6pm every day\" or \"We're open every day 10am-6pm\"")


if __name__ == '__main__':
    unittest.main()
