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

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# read in intents.json file
intents = json.loads(open('intents.json').read())

# load in the keys, and responseType from the pickle files and load the saved model
keys = pickle.load(open('keys.pk1', 'rb'))
responseType = pickle.load(open('responseType.pk1', 'rb'))
model = load_model('chatbotmodel.h5')


def constructSentence(sentence):
    """
    This is a function takes sentence and deconstructs it into its words, and breaks
    each word into it's stem word.
    Parameters:
            sentence (str): a sentence from user input
    Returns:
            sentenceWords (str): a sentence that only contains stem words for the model to use
    """
    sentenceWords = nltk.word_tokenize(sentence.lower())
    sentenceWords = [lemmatizer.lemmatize(word) for word in sentenceWords]
    return sentenceWords


def bagWords(sentence):
    """
    This is a function takes sentence and uses the constructSentence function and creates a
    bag of words (the same length as the keys) and constructs an array to match a response to
    this set of words
    Parameters:
            sentence (str): a sentence from user input
    Returns:
            bag (numpy array): an numpy array for the model
    """
    sentenceWords = constructSentence(sentence)
    bag = [0] * len(keys)

    for word in sentenceWords:
        # enumerate the list of keys
        for i, key in enumerate(keys):
            # if a key matches a word, set the bag at the given index to 1
            if key == word:
                bag[i] = 1

    return np.array(bag)


def predictResponse(sentence):
    """
    This is a function takes sentence and uses the bagWords function and predicts the responseType
    Parameters:
            sentence (str): a sentence from user input
    Returns:
            returnList (list): a list with responseType, and probability of that being the closest responseType
    """
    bow = bagWords(sentence)
    predictModel = model.predict(np.array([bow]))[0]
    # Specify the error threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(predictModel) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []
    for r in results:
        returnList.append({'intent': responseType[r[0]], 'probability': str(r[1])})
    return returnList


def getResponse(intentList, intentJSON):
    """
    This is a function takes the intentList and intentJSON, retrieves the tags from the JSON
    and checks if it matches the tags from intentList and chooses a random response
    (of the appropriate responses) to return to the user
    Parameters:
            intentList (list): a sentence from user input
            intentJSON (list): a json object
    Returns:
            result (list): a randomly selected response to the user input
    """
    tag = intentList[0]['intent']
    listIntents = intentJSON['intents']
    for i in listIntents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# run the chat bot
def main():
    print("Welcome, we are here to help you with your computer issues. Please type \"Hello\" "
          "or the type of issue you are having, to begin.")
    while True:
        userinput = input("Enter text: ")
        ints = predictResponse(userinput)
        res = getResponse(ints, intents)
        print("Agent: " + res)


if __name__ == '__main__':
    main()
