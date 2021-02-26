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

# Import models from tensor, layers and optimizers modules
from tensorflow.keras.models import Sequential
# Neural net layers
from tensorflow.keras.layers import Dense, Activation, Dropout
# Stochastic Gradient Descent optimization algorithm
from tensorflow.keras.optimizers import SGD

# Create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# read in intents.json file
intents = json.loads(open('intents.json').read())

# keys contains the list of all tags
keys = []
# responseType contains the list of responses
responseType = []
# keywords contains the list of patterns to look for in user input
keywords = []

# list containing punctuation to Ignore
ignoreChars = ['?', '!', '.', ',']

# iterate through the dictionary
for intent in intents['intents']:
    # find each list of patterns
    for pattern in intent['patterns']:
        # Split the sentence into individual words
        wordList = nltk.word_tokenize(pattern)
        # add each word to the keys list
        keys.extend(wordList)
        # add each list of wordlist and the associated tag to keywords
        keywords.append((wordList, intent['tag']))

        # print(keywords)
        if intent['tag'] not in responseType:
            # add tag to responseType
            responseType.append(intent['tag'])

# print(responseType)
# remove punctuation
keys = [lemmatizer.lemmatize(word) for word in keys if word not in ignoreChars]

# sort keys list
keys = sorted(set(keys))

# print(keys)
# sort responseType list
responseType = sorted(set(responseType))

# put keys and responseType in pickle files but writing in as binary
pickle.dump(keys, open('keys.pk1', 'wb'))
pickle.dump(responseType, open('responseType.pk1', 'wb'))


training = []
# create a empty list the same length of responseType
outputEmpty = [0] * len(responseType)

for keyword in keywords:
    # bag of words
    bag = []
    # retrieve the key at index 0
    wordPattern = keyword[0]
    # set all words to lower case
    wordPattern = [lemmatizer.lemmatize(word.lower()) for word in wordPattern]

    # if the key is in the wordPattern then add it to the bag
    for key in keys:
        bag.append(1) if key in wordPattern else bag.append(0)

    # copy the output list
    row = list(outputEmpty)
    # set all the indexes that correspond with a responseType to 1
    row[responseType.index(keyword[1])] = 1
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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# training the model 200 times, and save the model as an '.h5' model and print done
chatbot = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', chatbot)
print("Done")
