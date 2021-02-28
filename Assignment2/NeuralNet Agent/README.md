# Assignment 2
Computer science 310 Team 7

This project is on a conversational agent that takes word or sentence input from a user and outputs an appropriate response. Our specific conversational agent is a computer tech support agent that answers hardware and software related issues. The program was written in Python.

The Neural Net uses a JSON file as a database and each entry contains a tag (the conversation category), a list of patterns (individually typed messages on what the user is likely to type), and a list of random responses the agent will pick at random to use. Training the Neural Net takes this data from the JSON and essentially matches the tags to the patterns and responses, it deconstructs the patterns into bags of words to match with inputs later, and it stores all this in a binary file for fast easy execution of the program. The Neural Net has a more sophisticated portion where it finds consistently appearing unique words in the patterns of a tag and it assigns a higher probability to this tag if an input matches with this unique word.  

When using the Neural Net, it gets some input from the user, converts the input into a bag of words and rids of uppercases and punctuation, compares the bag of input words to the bags of patterns, finds patterns with the most word matches and gets the most probable tag, then it picks a response from the list in the tag. 

## Setting up Neural Net Agent
* Clone 'Assignment2' onto PC  
* Open the 'NeuralNet Agent' folder in a Python IDE  
* In terminal enter: 
  > ```pip install nltk```  
  > ```pip install tensorflow```  
  > ```pip install pandas```  
  > ```pip install numpy```  
  > ```pip install pickle```
* In python terminal enter
  > ```import nltk```  
  > ```nltk.download()```  
  > ```nltk.download('wordnet')```  
  > ```nltk.download('punkt')```
* Run '.py'  

## Compiling Neural Net
* Compile train.py (Only have to do this once, unless changes are made to the intents.json)
* **Note**: Do not be concerned with errors thrown in command console, there are some issues with the tensorflow library that do not affect the chatbot.

## Running Neural Net
* Compile agent.py
* **Note**: Do not be concerned with errors thrown in command console, there are some issues with the tensorflow library that do not affect the chatbot.

## List of Files
* **agent.py** *Runs the conversation agent program and takes in inputs to speak to it*
* **train.py** *Compiles the data from the intents.json*
* **intents.json** *Database that stores tags, corresponding patterns, corresponding responses*
* **keys.pk1** *Stores the character stream of tags to be reconstructed later for the agent script*
* **responseType.pk1** *Stores the character stream of responses to be reconstructed later for the agent script*
* **chatbotmodel.h5** *Trains the tags to have higher probabilities for certain words that consistently appear in its patterns and stores this information as a hierarchical data structure*
##  Imports 
* Random
* JSON 
* Pickle
* NumPy
* NLTK
* TensorFlow