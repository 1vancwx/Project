#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
from pandas import json_normalize

df = pd.read_json('intents.json')
df1 = pd.json_normalize(df['intents'])

df1['c_patt'] = [','.join(map(str,l)) for l in df1['patterns']]
df1['c_resp'] = [','.join(map(str,l)) for l in df1['responses']]
df1['combine'] = df1['c_resp']+df1['c_patt']+df1['tag']
df1['sentimental'] = ['negative' if 'sorry' in i.lower() else 'positive' for i in df1['combine']]
data = df1.drop(columns=['tag','patterns','responses','context','c_patt','c_resp'], axis = 1)
new_data = [{'combine':'Please connect me to the customer service', 'sentimental':'negative'},{'combine':'this is very useless', 'sentimental':'negative'}]
data = data.append(new_data, ignore_index=True)
df1


# In[2]:


import numpy as np
import nltk
nltk.download('wordnet')
import keras, string, re, html, math


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report

def cleaning(data):
    clean = re.sub('<.*?>', ' ', str(data))            #removes HTML tags
    clean = re.sub('\'.*?\s',' ', clean)               #removes all hanging letters afer apostrophes (s in it's)
    clean = re.sub(r'http\S+',' ', clean)              #removes URLs
    clean = re.sub('\W+',' ', clean)                   #replacing the non alphanumeric characters
    return html.unescape(clean)
data['cleaned'] = data['combine'].apply(cleaning)

def tokenizing(data):
    review = data['cleaned']                            #tokenizing is done
    tokens = nltk.word_tokenize(review)
    return tokens
data['tokens'] = data.apply(tokenizing, axis=1)

stop_words = set(stopwords.words('english'))
def remove_stops(data):
    my_list = data['tokens']
    meaningful_words = [w for w in my_list if not w in stop_words]           #stopwords are removed from the tokenized data
    return (meaningful_words)
data['tokens'] = data.apply(remove_stops, axis=1)

lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
    my_list = data['tokens']
    lemmatized_list = [lemmatizer.lemmatize(word) for word in my_list]    #lemmatizing is performed. It's more efficient and better than stemming.
    return (lemmatized_list)
data['tokens'] = data.apply(lemmatizing, axis=1)

def rejoin_words(data):
    my_list = data['tokens']
    joined_words = ( " ".join(my_list))                     #rejoins all stemmed words
    return joined_words
data['cleaned'] = data.apply(rejoin_words, axis=1)

data.head()


# In[3]:


def sents(data):
    clean = re.sub('<.*?>', ' ', str(data))            #removes HTML tags
    clean = re.sub('\'.*?\s',' ', clean)               #removes all hanging letters afer apostrophes (s in it's)
    clean = re.sub(r'http\S+',' ', clean)              #removes URLs
    clean = re.sub('[^a-zA-Z0-9\.]+',' ', clean)       #removes all non-alphanumeric characters except periods.
    tokens = nltk.sent_tokenize(clean)                 #sentence tokenizing is done
    return tokens
sents = data['combine'].apply(sents)

length_s = 0
for i in range(data.shape[0]):
    length_s+= len(sents[i])
print("The number of sentences is - ", length_s)          #prints the number of sentences

length_t = 0
for i in range(data.shape[0]):
    length_t+= len(data['tokens'][i])
print("\nThe number of tokens is - ", length_t)           #prints the number of tokens

average_tokens = round(length_t/length_s)
print("\nThe average number of tokens per sentence is - ", average_tokens) #prints the average number of tokens per sentence

positive = negative = 0
for i in range(data.shape[0]):
    if (data['sentimental'][i]=='positive'):
        positive += 1                           #finds the proprtion of positive and negative sentiments
    else:
        negative += 1

print("\nThe number of positive examples are - ", positive)
print("\nThe number of negative examples are - ", negative)
print("\nThe proportion of positive sentiments to negative ones are - ", positive/negative)


# In[4]:


# gets reviews column from df
reviews = data['cleaned'].values

# gets labels column from df
labels = data['sentimental'].values

encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
data['encoded']= encoded_labels
print(data['encoded'].head())

# prints(enc.classes_)
encoder_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("\nThe encoded classes are - ", encoder_mapping)

labels = data['encoded']


# In[5]:


# Splits the data into train and test (80% - 20%). 
# Uses stratify in train_test_split so that both train and test have similar ratio of positive and negative samples.
train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, labels, test_size=0.2, random_state=42, stratify=labels)

# train_sentences, test_sentences, train_labels, test_labels
print("The training sentences are -",train_sentences, sep='\n\n')
print("\nThe test sentences are -",test_sentences, sep='\n\n')
print("\nThe training labels are -",train_labels, sep='\n\n')
print("\nThe test labels are -",test_labels, sep='\n\n')


# In[6]:


# Uses Count vectorizer to get frequency of the words
vectorizer = CountVectorizer(max_features = 3000)

sents_encoded = vectorizer.fit_transform(train_sentences)         #encodes all training sentences
counts = sents_encoded.sum(axis=0).A1
vocab = list(vectorizer.get_feature_names())


# In[7]:


# Uses laplace smoothing for words in test set not present in vocab of train set.
class MultinomialNaiveBayes:
  
    def __init__(self, classes, tokenizer):
      #self.tokenizer = tokenizer
      self.classes = classes
      
    def group_by_class(self, X, y):
      data = dict()
      for c in self.classes:                            
#grouping by positive and negative sentiments
        data[c] = X[np.where(y == c)]
      return data
           
    def fit(self, X, y):
        self.n_class_items = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = vocab                            
#using the pre-made vocabulary of 3000 most frequent training words

        n = len(X)
        
        grouped_data = self.group_by_class(X, y)
        
        for c, data in grouped_data.items():
          self.n_class_items[c] = len(data)
          self.log_class_priors[c]=math.log(self.n_class_items[c]/n)
#taking log for easier calculation
          self.word_counts[c] = defaultdict(lambda: 0)
          
          for text in data:
            counts = Counter(nltk.word_tokenize(text))
            for word, count in counts.items():
                self.word_counts[c][word] += count
                
        return self
    def laplace_smoothing(self, word, text_class):          #smoothing
      num = self.word_counts[text_class][word] + 1
      denom = self.n_class_items[text_class] + len(self.vocab)
      return math.log(num / denom)
      
    def predict(self, X):
        result = []
        for text in X:
          
          class_scores = {c: self.log_class_priors[c] for c in self.classes}

          words = set(nltk.word_tokenize(text))
          for word in words:
              if word not in self.vocab: continue

              for c in self.classes:
                
                log_w_given_c = self.laplace_smoothing(word, c)
                class_scores[c] += log_w_given_c
                
          result.append(max(class_scores, key=class_scores.get))

        return result


# In[8]:


MNB = MultinomialNaiveBayes(
    classes=np.unique(labels), 
    tokenizer=Tokenizer()
).fit(train_sentences, train_labels)

# Tests the model on test set and reports the Accuracy
predicted_labels = MNB.predict(test_sentences)
print("The accuracy of the MNB classifier is ", accuracy_score(test_labels, predicted_labels))
print("\nThe classification report with metrics - \n", classification_report(test_labels, predicted_labels))


# In[9]:


oov_tok = '<OOK>'
embedding_dim = 100
max_length = 150
padding_type='post'
trunc_type='post'

# tokenizes sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sentences)

# vocabulary size
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1

# converts train dataset to sequence and pads sequences
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length)

# converts Test dataset to sequence and pads sequences
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)


# In[10]:


model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# compiles model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
model.summary()


# In[11]:


#training the model
num_epochs = 5
history = model.fit(train_padded, train_labels, 
                    epochs=num_epochs, verbose=1, 
                    validation_split=0.1)


# In[12]:


prediction = model.predict(test_padded)
print("The probabilities are - ", prediction, sep='\n')

# Gets labels based on probability 1 if p>= 0.5 else 0
for each in prediction:
    if each[0] >=0.5:
        each[0] = 1
    else:
        each[0] = 0
prediction = prediction.astype('int32') 
print("\nThe labels are - ", prediction, sep='\n')

# Calculates accuracy on Test data
print("\nThe accuracy of the model is ", accuracy_score(test_labels, prediction))
print("\nThe accuracy and other metrics are \n", classification_report(test_labels, prediction, labels=[0, 1]),sep='\n')


# In[13]:


sentence = ["The movie was very touching and heart whelming", 
            "I have never seen a terrible movie like this", 
            "the movie plot is terrible but it had good acting"]

# converts to a sequence
test_sequences = tokenizer.texts_to_sequences(sentence)

# pads the sequence
test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)

# Gets probabilities
prediction = model.predict(test_padded)
print("The probabilities are - ", prediction, sep='\n')

# Gets labels based on probability 1 if p>= 0.5 else 0
for each in prediction:
    if each[0] >=0.5:
        each[0] = 1
    else:
        each[0] = 0
prediction = prediction.astype('int32') 
print("\nThe labels are - ", prediction, sep='\n')


# In[ ]:




