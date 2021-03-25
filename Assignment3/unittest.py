import unittest
from Projects.Assignment3.NeuralNet_Agent.agent import *
from Projects.Assignment3.NeuralNet_Agent.train import *


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.agent = Agent()
        self.model = Model()

    def test_deconstructSentence(self):
        test_String = "What hours are you open?"
        self.assertEqual(self.agent.deconstructSentence(test_String),
                         ['what', 'hour', 'are', 'you', 'open', '?'], "Should be ['what', 'hour', 'are', 'you', "
                                                                      "'open', '?']")

    def test_getResponse(self):
        test_String = "What hours are you open?"
        predictedResponse = self.agent.predictResponse(test_String)
        self.assertTrue(self.agent.getResponse(predictedResponse) in ["Our hours are 10am-6pm every day",
                                                                      "We're open every day 10am-6pm"],
                        "Should be \"Our hours are 10am-6pm every day\" or \"We're open every day 10am-6pm\"")

    def test_synonyms1(self):
        test_string = "I am having software problems"
        synonym_set = {'software', 'problem', 'trouble'}
        self.assertEqual(self.model.synonyms(test_string), synonym_set)

    def test_synonyms2(self):
        test_string = "My pc is on the carpet"
        synonym_set = {'personal_computer', 'rug', 'carpet'}
        self.assertEqual(self.model.synonyms(test_string), synonym_set)

    def test_synonyms3(self):
        test_string = "The display looks weird"
        synonym_set = {'display', 'eldritch', 'weird', 'uncanny', 'unearthly'}
        self.assertEqual(self.model.synonyms(test_string), synonym_set)


if __name__ == '__main__':
    unittest.main()
