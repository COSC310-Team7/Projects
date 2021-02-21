import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

keys = []
responseType = []
keywords = []

ignoreChars = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        keys.extend(wordList)
        keywords.append((wordList, intent['tag']))

        # print(keywords)
        if intent['tag'] not in responseType:
            responseType.append(intent['tag'])

# print(responseType)
keys = [lemmatizer.lemmatize(word) for word in keys if word not in ignoreChars]
keys = sorted(set(keys))

# print(keys)

responseType = sorted(set(responseType))

pickle.dump(keys, open('keys.pk1', 'wb'))
pickle.dump(responseType, open('responseType.pk1', 'wb'))

training = []
outputEmpty = [0] * len(responseType)

for keyword in keywords:
    bag = []
    wordPattern = keyword[0]
    wordPattern = [lemmatizer.lemmatize(word.lower()) for word in wordPattern]
    for key in keys:
        bag.append(1) if key in wordPattern else bag.append(0)

    row = list(outputEmpty)
    row[responseType.index(keyword[1])] = 1
    training.append([bag, row])

random.shuffle(training)
training = np.array(training)

trainX = list(training[:, 0])
trainY = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(trainY[0]), activation="softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

chatbot = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', chatbot)
print("Done")
