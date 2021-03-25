# Module imports
# Random module
import random
# Json module
import json
# Pickle module
import pickle
# Numpy Module
import numpy as np
# Natural language tool kit module
import nltk
# Natural language tool kit stem module
from nltk.stem import WordNetLemmatizer

# Tensorflow models module to load in the model we trained
from tensorflow.keras.models import load_model

# Autocorrect -> spell check
from autocorrect import Speller


class Agent:
    """
        The class contains will contain the model that the chat bot will use to determine a response to user input.

        Attributes:
            lemmatizer (object): A wordnet lemmatizer object.
            intents (object): A JSON object containing all the structure of the neural net model.
            tags (list): A list containing all the response type tags from a pickle object.
            responses (list): A list containing all the responses from a pickle object.
            model (object): An object containing a trained model.

        Methods:
            deconstructSentence(): deconstructs sentences into their root words.
            spellCheck(): takes a sentence and corrects any spelling mistakes based on closest known word
            bagWords(): uses the deconstructed sentence to a series of words and maps it to a matching tag.
            predictResponse(): uses the chatbot model to return a response, with an associated probability.
            getResponse(): returns random bot response that has a greater probability than the minimum threshold.
            run(): runs the chatbot.
        """

    # Object constructor
    def __init__(self):
        # Create a lemmatizer object
        self.lemmatizer = WordNetLemmatizer()
        # read in intents.json file
        path = 'P:/COSC310 - Software Engineering/Projects/Projects/Assignment3/NeuralNet_Agent/'
        file = open(path + 'intents.json')
        self.intents = json.loads(file.read())
        file.close()
        # load in the tags, and responses from the pickle files and load the saved model
        file = open(path + 'tags.pk1', 'rb')
        self.tags = pickle.load(file)
        file.close()
        file = open(path + 'responses.pk1', 'rb')
        self.responses = pickle.load(file)
        file.close()
        self.model = load_model(path + 'chatbotmodel.h5')
        self.check = Speller(lang='en')

    def spellCheck(self, sentence):
        """
        This method takes a sentence and corrects any spelling mistakes based on closest known word
        Parameters:
            sentence (str): a sentence of user input
        Returns:
            corrected (str): a spell corrected sentence
        """
        corrected = self.check(sentence)
        return corrected

    def deconstructSentence(self, sentence):
        """
        This is a methods takes sentence and deconstructs it into its words, and breaks
        each word into it's stem word.
        Parameters:
            sentence (str): a sentence from user input
        Returns:
            separatedWords (list): a list containing individual root words of a given sentence
        """
        separatedWords = nltk.word_tokenize(sentence.lower())
        # print(separatedWords)
        separatedWords = [self.lemmatizer.lemmatize(word) for word in separatedWords]
        # print(separatedWords)
        return separatedWords

    def bagWords(self, sentence):
        """
        This is a methods takes sentence and uses the deconstructSentence methods and creates a
        bag of words (the same length as the tags), that is, it constructs an array with zeros everywhere
        except where the tags matches a word from the sentence.
        Parameters:
            sentence (str): a sentence from user input
        Returns:
            bag (numpy array): an numpy array for the model
        """
        separatedWords = self.deconstructSentence(sentence)
        bag = [0] * len(self.tags)

        # each word in the list
        for word in separatedWords:
            # enumerate the list of tags
            for (i, key) in enumerate(self.tags):
                # print("This is the key:", key)
                # if a key matches a word, set the bag at the given index to 1
                if key == word:
                    bag[i] = 1
        return np.array(bag)

    def predictResponse(self, sentence):
        """
        This is a methods takes sentence and uses the bagWords methods and predicts the responses
        Parameters:
            sentence (str): a sentence from user input
        Returns:
            potentialResponses (list): a list with responses, and probability of that being the closest responses
        """
        bow = self.bagWords(sentence)
        predictionModel = self.model.predict(np.array([bow]))[0]
        # Specify the error threshold
        ERROR_THRESHOLD = 0.25
        predictedResponses = [[i, r] for i, r in enumerate(predictionModel) if r > ERROR_THRESHOLD]

        predictedResponses.sort(key=lambda x: x[1], reverse=True)
        potentialReponses = []
        for r in predictedResponses:
            potentialReponses.append({'intent': self.responses[r[0]], 'probability': str(r[1])})
        # print(potentialReponses)
        return potentialReponses

    def getResponse(self, userSentence):
        """
        This is a methods takes the user input, retrieves the tags from the JSON
        and checks if it matches the tags from intents list and chooses a random response
        (of the appropriate responses) to return to the user
        Parameters:
            userSentence (list): a sentence from user input
        Returns:
            idealResponse (list): a randomly selected response to the user input
        """
        tag = userSentence[0]['intent']
        intents = self.intents['intents']
        for group in intents:
            if group['tag'] == tag:
                idealResponse = random.choice(group['responses'])
                break
        return idealResponse

    def run(self):
        """
        This methods receives user input, and uses the predictResponse methods to determine what the user's intention
        is, then uses the getResponse methods to determine an ideal response to return.
        """
        print("Welcome, we are here to help you with your computer issues. Please type \"Hello\" "
              "or the type of issue you are having, to begin.")
        while True:
            userInput = input("Enter text: ")
            correctedInput = self.spellCheck(userInput)
            print(correctedInput)

            if correctedInput.lower() == 'quit':
                break
            intentions = self.predictResponse(correctedInput)
            botResponse = self.getResponse(intentions)
            print("Agent: " + botResponse)


# run the chat bot
def main():
    chatBot = Agent()
    chatBot.run()


if __name__ == '__main__':
    main()
