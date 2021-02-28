# Importing modules
# Regular expressions operations module
import re
# Abstract Syntax Trees module
import ast
# Natural language tool kit module
from nltk.corpus import wordnet as wn

# Parse list of responses from separate text file
file = open("responses.txt", "r")
# Read line by line
contents = file.readlines()
dictionary = {}

# Structure of responses are as follows:
#       key : phrase
# Split each line at the colon
# Add to dictionary
for word in contents:
    (key, phrase) = word.split(":")
    # Ensure to remove any unnecessary characters
    dictionary[key] = phrase.strip("\n")

file.close()

# Testing purposes
print(dictionary)

# Parse the list of keywords from a separate text file
file = open("keywords.txt", "r")
contents = file.readlines()
list_syn = {}

for word in contents:
    synonyms = []
    word = word.strip("\n")
    for syn in wn.synsets(word):
        for lem in syn.lemmas():
            # Remove special characters
            lem_name = re.sub('[^a-zA-Z0-9 \n]', ' ', lem.name())
            synonyms.append(lem_name)
    list_syn[word] = set(synonyms)

file.close()

print(list_syn)


