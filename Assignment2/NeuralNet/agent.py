import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import pandas as pd
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

keys = pickle.load(open('keys.pk1', 'rb'))
responseType = pickle.load(open('responseType.pk1', 'rb'))
model = load_model('chatbotmodel.h5')

def constructSentence(sentence):
    sentenceWords = nltk.word_tokenize(sentence)
    sentenceWords = [lemmatizer.lemmatize(word) for word in sentenceWords]
    return sentenceWords

def bagWords(sentence):
    sentenceWords = constructSentence(sentence)
    bag = [0] * len(keys)

    for word in sentenceWords:
        for i, key in enumerate(keys):
            if key == word:
                bag[i] = 1

    return np.array(bag)

def predictResponse(sentence):
    bow = bagWords(sentence)
    predictModel = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(predictModel) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    returnList = []
    for r in results:
        returnList.append({'intent': responseType[r[0]], 'probability': str(r[1])})
    return returnList

def getResponse(intentList, intentJSON):
    tag = intentList[0]['intent']
    listIntents = intentJSON['intents']
    for i in listIntents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

while True:
    userinput = input("Enter text:")
    ints = predictResponse(userinput)
    res = getResponse(ints, intents)
    print(res)

