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
# Natural language tool kit corpus module -> for synonyms
from nltk.corpus import wordnet


# Import models from tensor, layers and optimizers modules
from tensorflow.keras.models import Sequential
# Neural net layers
from tensorflow.keras.layers import Dense, Activation, Dropout
# Stochastic Gradient Descent optimization algorithm
from tensorflow.keras.optimizers import SGD

# spaCy Natural language processing -> for pos tagging
import spacy
# Import english model
sp = spacy.load('en_core_web_sm')


class Model:
    """
    The class contains will contain the model that the chat bot will use to determine a response to user input.

    Attributes:
        lemmatizer (object): A wordnet lemmatizer object.
        intents (object): A JSON object containing all the structure of the neural net model.
        tags (list): A list containing all the response type tags from the intents object.
        responses (list): A list containing all the responses from the intents object.
        patterns (list): A list containing a sample of user inputs for a particular tag from the intents object.

    Methods:
        synonyms(str): returns a set of synonyms for nouns and adjectives in a given phrase
        train(): trains the bot using a Neural net.
    """

    # Object constructor
    def __init__(self):
        # Create a lemmatizer object
        self.lemmatizer = WordNetLemmatizer()
        # read in intents.json file
        path = 'P:/COSC310 - Software Engineering/Projects/Projects/Assignment3/NeuralNet_Agent/'
        self.intents = json.loads(open(path + 'intents.json').read())
        # tags contains the list of all tags
        self.tags = []
        # responses contains the list of responses
        self.responses = []
        # patterns contains the list of patterns to look for in user input
        self.patterns = []

    def synonyms(self, phrase):
        """
        Finds nouns and adjectives in a given phrase and gets the set of synonyms for these specific words

        Parameters
        phrase : string
            The phrase in which to find nouns and adjectives from

        Returns
        set
            A new set containing the synonyms of all nouns and adjectives of the input string

        Examples
        synonyms("I am having software problems")
        ['software', 'problem', 'trouble']
        """
        # Turn the string into an object with parts of speech tagging on each word
        sen = sp(u"" + phrase)
        # synonymset contains the set of synonyms
        synonymset = set()

        # iterate through the sentence
        for word in sen:
            # print(word, word.pos_)
            if word.pos_ == "NOUN":
                # print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
                # print(wordnet.synsets(word.text, pos=wordnet.NOUN))
                # iterate through synonyms for a noun (actually finds different concepts of the noun)
                for synonym in wordnet.synsets(word.text, pos=wordnet.NOUN):
                    # cleanup the synonym to get the name only and add it to the set of synonyms
                    partitioned_str = (synonym.name().partition('.'))
                    synonymset.add(partitioned_str[0])
            elif word.pos_ == "ADJ":
                # iterate through synonyms for an adjective (actually finds synonyms of each adj concept)
                for synonym in wordnet.synsets(word.text, pos=wordnet.ADJ):  # Each synset represents a diff concept.
                    synonymset.update(synonym.lemma_names())

        return synonymset

    def train(self):
        # list containing punctuation to Ignore
        ignoreChars = ['?', '!', '.', ',', '\'']
        # iterate through the dictionary
        for intent in self.intents['intents']:
            # find each list of patterns
            for pattern in intent['patterns']:
                # Split the sentence into individual words
                wordList = nltk.word_tokenize(pattern)
                # print(wordList)
                # Get synonyms of nouns and adjectives in wordlist
                synonymlist = self.synonyms(pattern)
                # Add each synonym to the word list
                # print(synonymlist)
                if synonymlist is not None:
                    wordList.extend(synonymlist)
                # print(wordList)
                # print(wordList)
                # print()
                # add each word to the tags list
                self.tags.extend(wordList)
                # add each list of wordlist and the associated tag to patterns
                self.patterns.append((wordList, intent['tag']))

                # print(patterns)
                if intent['tag'] not in self.responses:
                    # add tag to responses
                    self.responses.append(intent['tag'])

        # print(responses)
        # remove punctuation
        self.tags = [self.lemmatizer.lemmatize(word) for word in self.tags if word not in ignoreChars]

        # print(self.tags)
        # sort tags list
        self.tags = sorted(set(self.tags))

        # sort responses list
        self.responses = sorted(set(self.responses))

        # put tags and responses in pickle files but writing in as binary
        pickle.dump(self.tags, open('tags.pk1', 'wb'))
        pickle.dump(self.responses, open('responses.pk1', 'wb'))

        training = []
        # create a empty list the same length of responses
        outputEmpty = [0] * len(self.responses)

        for keyword in self.patterns:
            # bag of words
            bag = []
            # retrieve the key at index 0
            wordPattern = keyword[0]
            # set all words to lower case
            wordPattern = [self.lemmatizer.lemmatize(word.lower()) for word in wordPattern]

            # if the key is in the wordPattern then add it to the bag
            for key in self.tags:
                bag.append(1) if key in wordPattern else bag.append(0)

            # copy the output list
            row = list(outputEmpty)
            # set all the indexes that correspond with a responses to 1
            row[self.responses.index(keyword[1])] = 1
            # add this bag and row to the training list
            training.append([bag, row])

        # randomize the order of the training list
        random.shuffle(training)
        # convert this list to an array
        training = np.array(training)

        # split the list into X and Y values
        trainX = list(training[:, 0])
        trainY = list(training[:, 1])

        # Neural Net
        # Sequential model
        model = Sequential()

        # Layer 1 is a Dense layer that has 128 neurons with the same length as the input X value
        # and uses the rectified linear unit activation function

        model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
        # Layer 2 is a Dropout layer with a frequency rate of 0.5
        model.add(Dropout(0.5))

        # Layer 3 is a Dense layer that has 64 neurons and uses the rectified linear unit activation function

        model.add(Dense(64, activation='relu'))
        # Layer 4 is a Dropout layer with a frequency rate of 0.5
        model.add(Dropout(0.5))

        # Layer 5 is a Dense layer that has as many neurons as the length of the input Y values in the array
        # and uses the softmax activation function

        model.add(Dense(len(trainY[0]), activation="softmax"))

        # Stochastic gradient descent optimization algorithm with a learning rate of 0.01
        # decay of 0.000001, momentum of 0.9, and nesterov momentum set to true

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training the model 200 times, and save the model as an '.h5' model and print done
        chatbot = model.fit(np.array(trainX), np.array(trainY), epochs=300, batch_size=2, verbose=1)
        model.save('chatbotmodel.h5', chatbot)
        return "Done"


def main():
    model = Model()
    print(model.train())


if __name__ == "__main__":
    main()
