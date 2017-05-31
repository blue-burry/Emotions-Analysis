import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
import csv
from sklearn.ensemble import RandomForestClassifier

max_features = 200000
maxlen = 80
batch_size = 32

train=pd.read_csv('train_data.csv', sep=',', header=0)
test=pd.read_csv('test_data.csv', sep=',', header=0)


# splitting the data
features_train = train['content']
labels_train = train['sentiment']
features_test = test['content']


# processing sentiments
emotions_dictionary = {
                          'hate'    :   0,
                          'fun'     :   1,
                          'anger'   :   2,
                          'boredom' :   3,
                          'worry'   :   4,
                          'empty'   :   5,
                          'surprise':   6,
                          'love'    :   7,
                          'enthusiasm': 8,
                          'sadness' :   9,
                          'neutral' :   10,
                          'relief'  :   11,
                          'happiness':  12
}

labels_train = labels_train.map(emotions_dictionary)
combined = features_train.append(features_test).values
combined_features = []

# Tokenize, stem and remove stopwords from the combined data
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from string import punctuation

# stopwords
nltk.download('stopwords')
stop_words = list(set(stopwords.words('english')))
punc=list(set(punctuation))
stop_words.extend(punc)
stop_words.extend(["'s", "'d", "'m"])
print(stop_words)

for x in combined:
    x=word_tokenize(x)
    stemmer=SnowballStemmer('english')
    x=[(stemmer.stem(i)).lower() for i in x]
    x=[i for i in x if i not in stop_words]
    combined_features.append(x)

# mapping frequencies with words
from gensim import corpora
dictionary = corpora.Dictionary(combined_features)
print(dictionary)

id=[]
for x in combined_features:
    temp = [dictionary.token2id[j] for j in x]
    id.append(temp)

x_train=sequence.pad_sequences(np.array(id[:30000]))
    x_test=sequence.pad_sequences(np.array(id[30000:]))

y_train=labels_train

clf=RandomForestClassifier()
clf.fit(x_train, y_train)
preds=clf.predict(x_test)

inv_map = {v: k for k, v in emotions_dictionary.items()}

result=preds.map(inv_map)
result = pd.DataFrame({'sentiment':result})
result.index+=1
result.to_csv('result-3.csv', index_label='id')
