import pickle

import keras.src.saving
import numpy as np
import pandas as pd
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from tqdm import tqdm
from Sentiment_analysis_crypto import settings
import os
from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')


class CustomModel:
    def __init__(self, model_path, tokenizer):
        self.model = keras.models.load_model(model_path)
        self.max_words = 5000
        self.max_len = 50
        with open(tokenizer, 'rb') as f:
            self.tokenizer = pickle.load(f)

    @staticmethod
    def tweet2words(data):
        text = data.lower()
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        words = text.split()
        words = [w for w in words if w not in stopwords.words("english")]
        words = [PorterStemmer().stem(w) for w in words]
        return words

    def train_tokenizer(self):
        tokenizer = Tokenizer(num_words=self.max_words, lower=True, split=' ')
        df0 = pd.read_csv(os.path.join(settings.BASE_DIR.parent, 'Bitcoin_tweets_dataset_2.csv'), chunksize=100000,
                          lineterminator='\n')
        df = pd.concat(df0)
        cleantext = []
        df = df[['text']][0:2000]
        for item in tqdm(df['text']):
            words = CustomModel.tweet2words(item)
            cleantext += [' '.join(words)]
        df['cleantext'] = cleantext
        tokenizer.fit_on_texts(df['cleantext'])
        return tokenizer

    def tokenize_pad_sequences(self, text):
        X = self.tokenizer.texts_to_sequences(text)
        X = pad_sequences(X, padding='post', maxlen=self.max_len)
        return X, self.tokenizer

    def preprocessing(self, data):
        data = CustomModel.tweet2words(data)
        data, tokenizer = self.tokenize_pad_sequences([data])
        return data

    def give_pred(self, data):
        y_pred = self.model.predict(data)
        return np.argmax(y_pred)


model1 = CustomModel(os.path.join(settings.BASE_DIR, 'models', 'model1.keras'), os.path.join(settings.BASE_DIR, 'models', 'tokenizer.pkl'))
model2 = CustomModel(os.path.join(settings.BASE_DIR, 'models', 'model1.keras'), os.path.join(settings.BASE_DIR, 'models', 'tokenizer.pkl'))
