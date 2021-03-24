# Assignment 3
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
## Agent Class
The Agent class is located in the agent.py file. The Agent class has the following structure:
* Attributes:
  * ```lemmatizer (object): A wordnet lemmatizer object```
  * ```intents (object): A JSON object containing all the structure of the neural net model.```
  * ```tags (list): A list containing all the response type tags from a pickle object.```
  * ```responses (list): A list containing all the responses from a pickle object.```
  * ```model (object): An object containing a trained model.```
* Methods:
  * ```deconstructSentence(): deconstructs sentences into their root words.```
  * ```bagWords(): uses the deconstructed sentence to a series of words and maps it to a matching tag.```
  * ```predictResponse(): uses the chatbot model to return a response, with an associated probability.```
  * ```getResponse(): returns random bot response that has a greater probability than the minimum threshold.```
  * ```run(): runs the chatbot.```
  
## Model Class
The Model class is located in the train.py file. The Agent class has the following structure:
* Attributes:
  * ```lemmatizer (object): A wordnet lemmatizer object.```
  * ```intents (object): A JSON object containing all the structure of the neural net model.```
  * ```tags (list): A list containing all the response type tags from the intents object.```
  * ```responses (list): A list containing all the responses from the intents object.```
  * ```patterns (list): A list containing a sample of user inputs for a particular tag from the intents object.```
* Methods:
  * ```train(): trains the bot using a Neural net.```
### Note: Make sure you are using a version of python 3.8, python 3.9 has compatibility issues.

## ChatApplication Class (GUI)
The ChatApplication class is located in the app.py file. This class has the following strucure:
* Attributes:
  * ```window (object): A window object that holds the user interface.```
  * ```agent (object): An object that references the agent class as an object.```
* Methods:
  * ```run(): runs the chatbot in the GUI window mainloop.```
  * ```_setup_main_window(): A function of the window object that provides a title, window size and other features.```
  * ```_on_enter_pressed(): calls _insert_message function whenever user presses the enter button after typing a message.```
  * ```_insert_message(): takes a message and a sender as a parameter and inputs both message and response into the main message box. Note: calls on methods within agent.py as well.```


## Compile training data for the chatbot
* Compile train.py (Only have to do this once, unless changes are made to the intents.json)
* **Note**: Do not be concerned with errors thrown in command console, there are some issues with the tensorflow library that do not affect the chatbot.

## Running chatbot
* Compile agent.py
* **Note**: Do not be concerned with errors thrown in command console, there are some issues with the tensorflow library that do not affect the chatbot.

## List of Files
* **agent.py** *Runs the conversation agent program and takes in inputs to speak to it*
* **agent_unittest.py** *Runs a unit test on the Agent class*
* **train.py** *Compiles the data from the intents.json*
* **intents.json** *Database that stores tags, corresponding patterns, corresponding responses*
* **tags.pk1** *Stores the character stream of tags to be reconstructed later for the agent script*
* **responses.pk1** *Stores the character stream of responses to be reconstructed later for the agent script*
* **chatbotmodel.h5** *Trains the tags to have higher probabilities for certain words that consistently appear in its patterns and stores this information as a hierarchical data structure*
##  Imports 
* Random
* JSON 
* Pickle
* NumPy
* NLTK
* TensorFlow

## List of features
Each features that will be mentioned below will include a rationale as to why it has been chosen and a snippet of the feature in action.

### GUI
Simple GUI developed to run the program where user can view converstation history. This allows for a cleaner interaction.
