from gensim.models.ldamodel import LdaModel
from gensim.models import HdpModel
import roman
import pickle
import os
import numpy as np
from pprint import pprint
from itertools import chain
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.backend import dropout, truncated_normal
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import TensorBoard
import re
import tqdm
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer
import tensorflow as tf


#                                    #
#----------- Instructions -----------#
#                                    #

'''
To use this script (Sorry I didn't use notebooks), all you need to do is 
make sure you have a directory named checkpoints in the same folder as this, and 
the shakespeare sonnet data. Evertime you run this script, the dataset is loaded, 
preproccessed, the model is made, And then you will be prompted: if you would like 
to train the model or not (load a saved model). You may say No after the initial training.


You may change training values as you see fit, I have included my most recent training experiment.

To generate sentences, I have two functions. The first of the two functions Compare_sampling_methods()
takes in a prompt of any size, and compares two sampling techniques I found. They might be the same thing, 
written in different ways, but I wasn't sure so I wrote function. The second function incorporates the topic model as well.
generates_sentence_wLDA() takes in a prompt of any size, I haven't tested <3 length sentences. It adds the probability
from the topic model to each word as it is output from the softmax layer (Probably isn't the best way to do this).

You can uncomment the function calls I have right below the functions to see how they work.

'''



#                                    #
#------ Pre Proccesing of Data ------#
#                                    #
#----Create Dataset and save it
if not (os.path.exists('sonnets_data.pickle')):
    poem_number = 1
    with open('ern2017-session4-shakespeare-sonnets-dataset.txt') as sonnets:
        sonnets = sonnets.read().lower().strip().split('\n')
    data = []
    tmp =[]
    
    for curr_line in range(len(sonnets)):
        line = sonnets[curr_line].strip()
        if line == roman.toRoman(poem_number).lower() :
            poem_number+=1
            data.append(tmp)
            tmp=[]
            print(line) if curr_line == 2 else 0
        else:  
            #This line just removes all punctuation, found LDA to preform much better
            line=re.sub(r'[^\w\s]','',line)
            tmp.append(line)
        curr_line+=1
    data.append(tmp)
    data= data[1:]
    with open('sonnets_data.pickle','wb') as pickle_out:
        pickle.dump(data,pickle_out)
else:
    with open('sonnets_data.pickle','rb') as pickle_in:
        data = pickle.load(pickle_in)


# Function that takes in a complete sentence and returns an n-gram version of it
def returnNgram(sentence,N,isTokenized=True):
    #Makes sure we use Tokenized sentence
    if not(isTokenized): sentence = sentence.split()
    ngrammed = [sentence[word_index:word_index+N] for word_index in range(len(sentence)-N+1)]
    return ngrammed

# Takes a sentence and generates every possible Ngram
def variableNgram(sentence):
    variableNgrammed=[]
    #Generates at least a trigram which turns into a bigram+label
    #Makes sure Input isn't too long
    max_len = len(sentence) if len(sentence)<=50 else 50
    for i in range(3,max_len):
        variableNgrammed.extend(returnNgram(sentence,i))
    return variableNgrammed

# ... Pads the dataset.
def padDataset(dataset,pad_len=False):
    if not(pad_len):
        data_lens = [len(line) for line in dataset]
        max_len = max(data_lens)
        dataset=[[0]*(max_len-len(sent)) + sent for sent in dataset]
        return max_len,np.array(dataset)
    elif isinstance(pad_len,int):
        dataset=[[0]*(pad_len-len(sent)) + sent for sent in dataset]
        return dataset
# print(data[0:2])
#----- Tokenization
flattened_data=list(chain.from_iterable(data))
tokens = Tokenizer()

tokens.fit_on_texts(flattened_data)
total_words=len(tokens.word_index)+1

#Tokenizes flattened_data
tokenized_dataset=tokens.texts_to_sequences(flattened_data)

#----- Padding
# Genereated Every possible NGramm for each sentence
# flattened_Ngrammed_data = []
# for sentence in tokenized_dataset:
#     flattened_Ngrammed_data.extend(returnNgram(sentence,4))
# input_len,flattened_Ngrammed_padded_data= padDataset(flattened_Ngrammed_data)

input_len,flattened_padded_data=padDataset(tokenized_dataset)

x,labels=flattened_padded_data[:,:-1],flattened_padded_data[:,-1]
# Maybe make y into a list of words? 
y=to_categorical(labels,num_classes=total_words)
# 2155 Sentences, 3255 vocab

#                                                #
#--------- Build Embeddings/Topic model ---------#
#                                                #
def unused_exploratory_code():
    # inside a function so can be hidden in VSCODE via dropdown, can be deleteed
    def coherence_per_topic_val():
        '''
        START=10
        STOP=100
        STEP=5
        topic_range=range(START, STOP, STEP)
        coherence_values=[]
        model_list=[]
        for num_topic in topic_range:
            topic_model=LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topic,workers=4)
            model_list.append(topic_model)
            coherence_model=CoherenceModel(model=topic_model,texts=flattened_data,dictionary=id2word,coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())
        max_coherence_val=0
        optimal_model=None
        for i,(m,coherence) in enumerate(zip(topic_range,coherence_values)):
            if max_coherence_val<round(coherence,4):
                optimal_model=model_list[i]
                
                max_coherence_val=round(coherence,4)
            print("NUM TOPICS=",m," Has Coherence Value of ", round(coherence,4))'''
    def importGloveEmbeddings():
        '''
        # Loads in GloVe embeddings if they don't exist
        if not(os.path.exists('EmbeddingVectors.pickle')):
            EMBEDDING_VECTOR_LENGTH = 50 # <=200
            embedding_dict = {}
            with open('glove.42B.300d.txt',encoding='utf-8') as f:
                for line in tqdm.tqdm(f.readlines()):
                    values=line.split()
                    # get the word
                    word=values[0]
                    if word in tokens.word_index.keys():
                        # get the vector
                        vector = np.asarray(values[1:EMBEDDING_VECTOR_LENGTH], 'float32')
                        embedding_dict[word] = vector
            with open('EmbeddingVectors.pickle','wb') as pickle_out:
                pickle.dump(embedding_dict,pickle_out)
        else:
            with open('EmbeddingVectors.pickle','rb') as pickle_in:
                embedding_dict = pickle.load(pickle_in)'''

stop_words = stopwords.words('english')
stop_sequences=tokens.texts_to_sequences([stop_words])[0]
stop_words.extend(['thee','thee,','thus','thine','me,','i','though','two','thy','make','doth','thou','shall','hath','me','dost','would','know',
                    'not,','thee.'])

lemmatizer = WordNetLemmatizer()

# Creates tokenized dataset that is seperated by poems, each word is checked
# against the stop_words list and is lemmatized
documents_without_stopwords=[]
for poem in data:
    tmp=[]
    for line in poem:
        tmp.extend([lemmatizer.lemmatize(word) for word in line.split() if not(word in stop_words)])
    documents_without_stopwords.append(tmp)

id2word=Dictionary(documents_without_stopwords)
corpus = [id2word.doc2bow(text) for text in documents_without_stopwords]


topic_model=HdpModel(corpus=corpus,id2word=id2word)
alpha,beta=topic_model.hdp_to_lda()
#Trained a HDP model and used the num_topics, alpha and beta values as guidelines for a LDA model
lda_topic_model=LdaModel(corpus=corpus,id2word=id2word,num_topics=14,alpha=alpha[:14],eta=beta[:14])
# Not sure why Topics don't make as much sense as I would have liked
# pprint(lda_topic_model.print_topics())



#                                    #
#---------- Build The Model ---------#
#                                    #

def makeModel():
    model = Sequential()
    # How to decide the output dim of embedding?
    model.add(Embedding(total_words,500,input_length= input_len))
    model.add(Bidirectional(LSTM(500)))
    model.add(Dense(total_words,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    model.summary()
    return model
model = makeModel()



#                                    #
#---------- Train The Model ---------#
#                                    #
train =input('Should we Train? Y/N:')
train =True if train == "Y" else False


if os.path.exists(r'.\\checkpoints'):
    chckpoints = r'.\\checkpoints\\'
    checkpoint_callback=ModelCheckpoint(filepath=chckpoints+r'weights.{epoch:02d}.hdf5',
                                        save_weights_only=True,verbose=1)
    tensorboard_callback=TensorBoard()

    if not(train):
        path = os.listdir(chckpoints)[-1]
        print("Using this:",path)
        model.load_weights(chckpoints+path)
        model.compile(loss='categorical_crossentropy',
                    optimizer='Adam',metrics=['accuracy'])
    else:
        # path = os.listdir(chckpoints)[-1]
        # print("Using this:",path)
        # model.load_weights(chckpoints+path)
        model.compile(loss='categorical_crossentropy',
                    optimizer='Adam',metrics=['accuracy'])
    
        model.fit(x,y,epochs=10,batch_size=64,
                verbose=1,callbacks=[checkpoint_callback,tensorboard_callback])
else:
    raise Exception("Please Make Sure that a "+
                    "folder named Checkpoints is in the same directory")




#                                    #
#-------- Generate Sentences --------#
#                                    #

def sample(probs, temperature): 
    """samples an index from a vector of probabilities""" 
    probs = np.asarray(probs).astype('float64')
    a = np.log(probs)/temperature 
    a = np.exp(a)/np.sum(np.exp(a)) 
    return np.argmax(np.random.multinomial(1, a, 1)) 


def Compare_sampling_methods(prompt,length=10):
    np_sentence=prompt
    sample_sentence=prompt
    for i in range(length):
        returns=[]
        tokenized_np_sentence =tokens.texts_to_sequences(np_sentence)
        tokenized_sample_sentence=tokens.texts_to_sequences(sample_sentence)

        tokenized_padded_np = padDataset(tokenized_np_sentence,input_len)
        tokenized_padded_sample = padDataset(tokenized_sample_sentence,input_len)
        
        model_np = model.predict(tokenized_padded_np)
        model_sample=model.predict(tokenized_padded_sample)

        sample_predictions=[]
        np_predictions=[]
        
        words=list(tokens.word_index.keys())
        words.append('oov')

        for pred in model_np:
            prediction = np.random.choice(words, p = pred)
            np_predictions.append(prediction)
        for pred in model_sample:
            sample_predictions.append(sample(pred,0.9))    
        
        sample_predictions=tokens.sequences_to_texts([sample_predictions])
        np_sentence+=' '.join(np_predictions)
        sample_sentence+=' ' +sample_predictions[0]
        return np_sentence,sample_sentence

print(Compare_sampling_methods("From fairest creatures we desire"))

# Best results from SimpleRNN : 
# From fairest creatures we desire pretty feeds heavy heavy blessed seen time's under 
# From fairest creatures we desire grow pride hand argument grow grow grow argument  

#Best Results from one layer LSTM:
# From fairest creatures we desire puts cruel bear triumph triumph so victors varying 
# From fairest creatures we desire use me thee land hope favour thee store

    
#--- Generate sentence with topic models
def generates_sentence_wLDA(prompt,length=10):
    words=list(tokens.word_index.keys())
    words.append('oov')
    sample_sentence=prompt
    
    # Gets the topics in the sentence
    zipped=lda_topic_model.get_document_topics(id2word.doc2bow(sample_sentence.split()))
    unzipped=list(zip(*zipped))

    # Gets the highest probability topic
    max_topic_value=unzipped[1].index(max(unzipped[1]))
    max_topic=unzipped[0][max_topic_value]

    # Gets words in that topic. The words are Dictionary ids
    words_for_topic= lda_topic_model.get_topic_terms(max_topic)
    id_maps=id2word.id2token

    # Converts the ids of those words into keras.tokenizer tokens
    for i,id in enumerate(words_for_topic):
        words_for_topic[i]=[tokens.texts_to_sequences([id_maps[id[0]]])[0][0],words_for_topic[i][1]]
    # print(words_for_topic)

    for i in range(length):
        
        tokenized_sample_sentence=tokens.texts_to_sequences(sample_sentence)
        tokenized_padded_sentence = padDataset(tokenized_sample_sentence,input_len)
        model_np = model.predict(tokenized_padded_sentence)
        for word in words_for_topic:
            
            model_np[0][word[0]] = model_np[0][word[0]]+word[1]

        pred = [float(i)/sum(model_np[0]) for i in model_np[0]]
        prediction = np.random.choice(words, p = pred)
        sample_sentence+=" " +prediction
    return sample_sentence
print(generates_sentence_wLDA("From fairest creatures we desire"))

'''
#Remove stopwords
sample_sentence_stopwords=[word for word in sample_sentence.split() if word not in stop_words]
print(sample_sentence_stopwords)
# print(embedding_dict)
glove_vectors=[]
keys = list(embedding_dict.keys())
for word in sample_sentence_stopwords:
    if word in list(keys):
        glove_vectors.append(embedding_dict[word])
print(glove_vectors)'''

