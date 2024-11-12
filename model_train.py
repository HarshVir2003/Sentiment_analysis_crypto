import pandas as pd
import os
from tqdm import tqdm

# For Preprocessing
import re  # RegEx for removing non-letter characters
import nltk  # natural language processing

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# For Building the model
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import datasets
from tensorflow.keras import losses

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df0 = pd.read_csv(r'D:\Projects\Project_sentiment_analysis\Bitcoin_tweets_dataset_2.csv', chunksize=100000,
                  lineterminator='\n')
df = pd.concat(df0)


def tweet2words(tweet):
    '''
    Convert tweet text into a sequence of words
    :tweet -> text data
    '''

    # convert to lowercase
    text = tweet.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words


# df=df[['text']][0:2000]

cleantext = []
for item in tqdm(df['text']):
    words = tweet2words(item)
    cleantext += [words]
df['cleantext'] = cleantext


def unlist(list):
    words = ''
    for item in list:
        words += item + ' '
    return words


def compute_vader_scores(df, label):
    sid = SentimentIntensityAnalyzer()
    df["vader_neg"] = df[label].apply(lambda x: sid.polarity_scores(unlist(x))["neg"])
    df["vader_neu"] = df[label].apply(lambda x: sid.polarity_scores(unlist(x))["neu"])
    df["vader_pos"] = df[label].apply(lambda x: sid.polarity_scores(unlist(x))["pos"])
    df["vader_comp"] = df[label].apply(lambda x: sid.polarity_scores(unlist(x))["compound"])
    df['cleantext2'] = df[label].apply(lambda x: unlist(x))
    return df


nltk.download('vader_lexicon')
df2 = compute_vader_scores(df, 'cleantext')

class0 = []
for i in range(len(df2)):
    if df2.loc[i, 'vader_neg'] > 0:
        class0 += [0]
    elif df2.loc[i, 'vader_pos'] > 0:
        class0 += [2]
    else:
        class0 += [1]

df['class'] = class0

max_words = 5000
max_len = 50


def tokenize_pad_sequences(text):
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    '''
    # Text tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)
    # return sequences
    return X, tokenizer


print('Before Tokenization & Padding \n', df['cleantext2'][0])
X, tokenizer = tokenize_pad_sequences(df['cleantext2'])
print('After Tokenization & Padding \n', X[0])

y = pd.get_dummies(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print('Train Set: ', X_train.shape, y_train.shape)
print('Validation Set: ', X_val.shape, y_val.shape)
print('Test Set: ', X_test.shape, y_test.shape)


def f1_score(precision, recall):
    ''' Function to calculate f1 score '''

    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# hyper parameters
vocab_size = 5000
embedding_size = 32
epochs = 100
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8
import tensorflow as tf

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
sgd = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

from tensorflow.keras.optimizers import SGD

sgd = SGD(0.1, momentum=momentum, decay=decay_rate, nesterov=False)
# Build model
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', Precision(), Recall()])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=64, epochs=epochs, verbose=1)

# model.save(r'\models\model1')
model.save(r'model1.h5')
model.save(r'model1.keras')
