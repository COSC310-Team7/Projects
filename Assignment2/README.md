# Assignment 2
Computer science 310 Team 7

This project is on a conversational agent that takes word or sentence input from a user and outputs an appropriate response. Our specific conversational agent is a computer tech support agent that answers hardware and software related issues. The program specifically takes in an input, gets a list of synonyms of the input, matches one of the synonyms to our intent, then generates the appropriate response based on the intent. The program was written in Python and generating these synomyns were done with Natural Language Toolkit (nltk) and wordnet.

## Running and compiling
* Clone 'Assignment2' somewhere on your pc  
* Open the 'Assignment2' folder in a Python IDE  
* In a terminal enter 
  > pip install nltk  
* In python terminal enter
  > import nltk  
  > nltk.download()  
  > nltk.download('wordnet')  
* Run '.py'  

## Update to run Neural Net bot
* In terminal enter
  > pip install tensorflow
  > pip install pandas
  > pip install numpy
* May need to install a module from nltk
  > import nltk
  > nltk.download(' "specified package" ')

## Run and use Neural Net
* Compile trian.py is any updates are made to intents.json
* Compile agent.py
* **Note**: Do not be concerned with errors thrown in command console, there are some issues with the tensorflow library that do not affect the chatbot.
