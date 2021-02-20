# Module imports
# import trie structure module
# import pytrie
# Regular expressions operations module
import re
# Abstract Syntax Trees module
# import ast
# Natural language tool kit module
from nltk.corpus import wordnet as wn


class CallAgent:
    """
    This is a class for the call agent where the inputs and responses are compiled and generated

    Attributes:
        responses (dict): Dictionary of intent and corresponding response
        keywords (dict): Dictionary of intent and corresponding set of synonyms
    """
    responses = {}
    keywords = {}

    def __init__(self, responsefile, keywordfile):
        """
        The constructor for CallAgent class

        Parameters:
            responsefile (str): The name of the txt file containing intent:response pairs
            keywordfile (str): The name of the txt file containing keywords for synonyms to generate
        """
        self.generate_repsonses(responsefile)
        self.generate_keywords(keywordfile)

    def generate_repsonses(self, file):
        """
        The function to parse the response file and store intent and response pairs in a dict

        Parameters:
            file (str):  The name of the txt file containing intent:response pairs
        Sets Attribute:
            responses (dict): Dictionary of intent and corresponding response
        """
        # Parse list of responses from separate text file
        file = open(file, "r")
        # Read line by line
        contents = file.readlines()
        # Structure of responses are as follows:
        #       intent : response
        # Split each line at the colon
        # Add to dictionary
        for word in contents:
            (responseType, phrase) = word.split(":")
            # Ensure to remove any unnecessary characters
            self.responses[responseType] = phrase.strip("\n")
        file.close()

    def generate_keywords(self, file):
        """
        The function to parse the keyword file and generate synonyms for each keyword
        The 'keyword:synonym list' pairs are stored in a dict
        The keyword is replaced by its corresponding intent, making it 'intent:synonym list'

        Parameters:
            file (str): The name of the txt file containing keywords
        Sets Attribute:
            keywords (dict): Dictionary of intent and corresponding set of synonyms
        """
        # Parse list of responses from separate text file
        file = open(file, "r")
        # Read line by line
        contents = file.readlines()
        # Get all the intents that are mapped to responses
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
            # print(synonyms)
            if not synonyms:
                # print(words)
                words = re.sub('[^a-zA-Z0-9 \n]', ' ', words)
                synonyms.append(words)
                list_syn.append(set(synonyms))
                # print(list_syn)
            else:
                list_syn.append(set(synonyms))
        file.close()

        for i in range(0, len(key), 1):
            # Map each intent to the list of synonyms of a given keyword
            self.keywords[key[i]] = list_syn[i]

    # function to get the dictionary of responses
    def get_responses(self):
        """
        The function to return the dict of intent:response pairs

        Returns:
             responses (dict): Dictionary of intent and corresponding response
        """
        return self.responses

    # function to get dictionary of keywords
    def get_keywords(self):
        """
        The function to return the dict of 'intent:synonym list' pairs

        Returns:
             keywords (dict): Dictionary of intent and corresponding set of synonyms
        """
        return self.keywords

    def unused(self):
        """
        The function to take user input and iterate through the lists of synonyms to find a match
        A response is printed corresponding to its matching intent and synonym.

        user types 'quit' to stop
        """
        while True:
            # user enters a prompt
            userprompt = input("Enter some text:")
            userprompt = re.sub('[^a-zA-Z0-9 \n]', ' ', userprompt)
            # iterate through keywords
            for responseType in self.keywords:
                # iterate through synonyms associated with a keyword
                for syn in self.keywords[responseType]:
                    # print(syn)
                    # check to see if any of the synonyms are in the userprompt
                    if syn in userprompt.lower():
                        # exit if the user wants to quit
                        if responseType == "quit":
                            print(self.responses[responseType])
                            return
                        else:
                            # response with the phrase associated with that keyword
                            print(self.responses[responseType])

    def run(self):
        while True:
            return


def main():
    ca = CallAgent("responses.txt", "keywords.txt")
    # print(ca.get_responses())
    # print(ca.get_keywords())
    ca.unused()

if __name__ == "__main__":
    main()
