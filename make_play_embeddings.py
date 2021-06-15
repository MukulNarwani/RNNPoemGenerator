import urllib.request
import re
import glove
from glove import Corpus, Glove # creating a corpus object


# change to your own path if you have downloaded the file locally
url = 'https://dataskat.s3.eu-west-3.amazonaws.com/data/Shakespeare_alllines.txt'
# read file into list of lines
lines = urllib.request.urlopen(url).read().decode('utf-8').split("\n")

sentences = []
for line in lines:
    # remove punctuation
    line = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',line).strip()
    # tokenizer
    tokens = re.findall(r'\b\w+\b', line)
    if len(tokens) > 1:
        sentences.append(tokens)


# instantiate the corpus
corpus = Corpus() 

# this will create the word co occurence matrix 
corpus.fit(sentences, window=10)

# instantiate the model
glove = Glove(no_components=50, learning_rate=0.05)

# and fit over the corpus matrix
glove.fit(corpus.matrix, epochs=30, no_threads=2)

# finally we add the vocabulary to the model
glove.add_dictionary(corpus.dictionary)