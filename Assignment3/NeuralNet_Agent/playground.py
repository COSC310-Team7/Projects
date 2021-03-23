from nltk.corpus import wordnet
import json
import spacy
sp = spacy.load('en_core_web_sm')

intents = json.loads(open('intents.json').read())
alreadylist = []

# the implementation for nouns differ in that we use the name of each synset as the synonym
# what synset means it is a different concept or interpretation of the meaning of a word
# we actually do not use the actual list of synonyms for the word because of the many different synsets
# the following example is in the format:
# -------------------
# noun
# -------------------
# synset name (format: name.postag.#)
# synset definition
# synonyms of the synset
#
# synset name
# synset definition
# synonyms of the synset     etc...

# """ remove hashtag to block comment code
for intent in intents['intents']:
    # find each list of patterns
    for pattern in intent['patterns']:
        sen = sp(u"" + pattern)
        for word in sen:
            if ((word.pos_ == "NOUN") & (word.text not in alreadylist)):
                alreadylist.append(word.text)
                print("-----------------------------------------------")
                print(word.text)
                print("-----------------------------------------------")
                for ss in wordnet.synsets(word.text, pos=wordnet.NOUN):  # Each synset represents a diff concept.
                    print(ss.name())
                    print(ss.definition())
                    print(ss.lemma_names())
                    print()
# """

# the implementation for adjectives differ in that we use synonym list of every synset
# what synset means it is a different concept or interpretation of the meaning of a word
# we do this because it makes slightly more sense for adjectives
# the following example is in the format:
# -------------------
# adjective
# -------------------
# synset name (format: name.postag.#)
# synset definition
# synonyms of the synset
#
# synset name
# synset definition
# synonyms of the synset     etc...

# """ remove hashtag to block comment code
alreadylist = []
for intent in intents['intents']:
    # find each list of patterns
    for pattern in intent['patterns']:
        sen = sp(u"" + pattern)
        for word in sen:
            if ((word.pos_ == "ADJ") & (word.text not in alreadylist)):
                alreadylist.append(word.text)
                print("-----------------------------------------------")
                print(word.text)
                print("-----------------------------------------------")
                for ss in wordnet.synsets(word.text, pos=wordnet.ADJ):  # Each synset represents a diff concept.
                    print(ss.name())
                    print(ss.definition())
                    print(ss.lemma_names())
                    print()
# """
"""
for ss in wordnet.synsets('software'): # Each synset represents a diff concept.
    print(ss.name())
    print(ss.definition())
    print(ss.lemma_names())
    print()
"""
"""
sen = sp(u"what hours are you open")
for word in sen:
    if word.pos_ == "ADJ":
        print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
        for ss in wordnet.synsets(word.text, pos=wordnet.ADJ):  # Each synset represents a diff concept.
            print(ss.name())
            print(ss.definition())
            print(ss.lemma_names())
        #synonlist = set()
        #for synon in wordnet.synsets(word.text, pos=wordnet.NOUN):
        #    partitioned_str = (synon.name().partition('.'))
         #   synonlist.add(partitioned_str[0])
        #print(synonlist)
        print()
"""
"""
print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
print(wordnet.synsets(word.text, pos=wordnet.NOUN))
synonlist = set()
for synon in wordnet.synsets(word.text, pos=wordnet.NOUN):
    partitioned_str = (synon.name().partition('.'))
    synonlist.add(partitioned_str[0])
print(synonlist)
print()
"""