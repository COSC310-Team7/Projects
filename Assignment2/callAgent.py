# Module imports
# import trie structure module
import pygtrie
# Regular expressions operations module
import re
# Abstract Syntax Trees module
import ast
# Natural language tool kit module
from nltk.corpus import wordnet as wn


class CallAgent:
    # response - list of response-types and corresponding phrases
    # keywords - list of response-types mapped to keywords
    responses = {}
    keywords = {}

    def __init__(self, responsefile, keywordfile):
        self.generate_repsonses(responsefile)
        self.generate_keywords(keywordfile)

    def generate_repsonses(self, file):
        # Parse list of responses from separate text file
        file = open(file, "r")
        # Read line by line
        contents = file.readlines()
        # Structure of responses are as follows:
        #       response-type : phrase
        # Split each line at the colon
        # Add to dictionary
        for word in contents:
            (key, phrase) = word.split(":")
            # Ensure to remove any unnecessary characters
            self.responses[key] = phrase.strip("\n")
        file.close()

    def generate_keywords(self, file):
        # Parse list of responses from separate text file
        file = open(file, "r")
        # Read line by line
        contents = file.readlines()
        # Get all the response-types that are mapped to responses
        key = list(self.get_responses().keys())
        # Create an empty list that will hold all synonyms of a given keyword
        list_syn = []

        for words in contents:
            synonyms = []
            words = words.strip("\n")
            for syn in wn.synsets(words):
                for lem in syn.lemmas():
                    # Remove special characters
                    lem_name = re.sub('[^a-zA-Z0-9 \n]', ' ', lem.name())
                    synonyms.append(lem_name)
            # Add all the synonyms to the list
            list_syn.append(set(synonyms))
        file.close()

        for i in range(0, len(key), 1):
            # Map each response-types to the list of synonyms of a given keyword
            self.keywords[key[i]] = list_syn[i]

    # function to get the dictionary of responses
    def get_responses(self):
        return self.responses

    # fucntion to get dictionary of keywords
    def get_keywords(self):
        return self.keywords

    def run(self):
        while True:
            # user enters a prompt
            userprompt = input("Enter some text:")
            # iterate through keywords
            for key in self.keywords:
                # iterate through synonyms associated with a keyword
                for syn in self.keywords[key]:
                    # check to see if any of the synonyms are in the userprompt
                    if syn in userprompt.lower():
                        # exit if the user wants to quit
                        if key == "quit":
                            print(self.responses[key])
                            return
                        else:
                            # response with the phrase associated with that keyword
                            print(self.responses[key])


def main():
    ca = CallAgent("responses.txt", "keywords.txt")
    # print(ca.get_responses())
    # print(ca.get_keywords())
    ca.run()


if __name__ == "__main__":
    main()
