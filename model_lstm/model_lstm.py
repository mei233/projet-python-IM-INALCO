#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 19:17:40 2021

@author: mei
"""
'''
Ce script est de l'objectif d'entraîner le classifieurs de lstm avec word embedding sur les commentaiers chinois des plusieurs catégories
'''

#coding=utf-8
import pandas as pd
import matplotlib
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import jieba 
from keras.models import load_model
import joblib
import re
from  sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

#delect all the letters, numbers and puntuations
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

 #  delect stopwords
def stopwordslist(filepath): 
    stopwords = []
    for line in open(filepath, 'r', encoding='utf-8').readlines():
        liste = line.strip().split(',')
    stopwords.extend(liste)  
    return stopwords  

stopwords = stopwordslist("chineseStopWords.txt")
    
class Model_LSTM:
    def __init__(self):
        self.MAX_NB_WORDS = 50000
        self.MAX_SEQUENCE_LENGTH = 250
        self.EMBEDDING_DIM = 100
        
        
        
    def save(self):
        '''
        save the model

        Returns
        -------
        None.

        '''
        print('Saving model...\n')
        self.lstm_model.save('lstm_model.h5')
        print('model saved as lstm_model.h5')
        print('Saving tokenizer...\n')
        joblib.dump(self.tokenizer, "lstm_tokenizer.m")
        print('model saved as lstm_tokenizer.m')
        
    
    def load_model(self,model_file,tokenizer_file):
        '''
    
        Parameters
        ----------
        model_file : 
            the path and the file name of the model file

        Returns
        -------
        None.

        '''
        print(f"Loading model {model_file}...")
        self.lstm_model = load_model(model_file)
        
        print(f"Loading model {tokenizer_file}...")
        self.tokenizer = joblib.load(tokenizer_file) 
        
    
    def pretraitement(self):
        '''
        pretreatement of the corpus

        Returns
        -------
        None.

        '''
        
        self.df = pd.read_csv('corpus.csv')
        # convert cat into cat_id
        self.df = self.df.drop_duplicates(subset=['review'], keep='first', inplace=False)
        self.df['cat_id'] = self.df['cat'].factorize()[0]
        self.cat_id_df = self.df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
        # cat_to_id = dict(cat_id_df.values)
        
        # self.df['clean_review'] = self.df['review'].apply(remove_punctuation)
        #  segmentation and delect stopwords
        self.df['cut_review'] = self.df['review'].astype(str).apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
        self.df['length'] = self.df['cut_review'].str.split().str.len()
        self.df = self.df[self.df['length'] >= 5]
    
    
    
    
    def pipeline(self,save_model=False):
        '''
        training the model

        Parameters
        ----------
        save_model : TYPE, optional
            DESCRIPTION. The default is False.
            if save_model = True : save the trainning model(a lstm trainning modele and a vectorizer model)

        Returns
        -------
        None.

        '''
        self.pretraitement()
        self.tokenizer = Tokenizer(num_words = self.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        self.tokenizer.fit_on_texts(self.df['cut_review'].values)
        # word_index = tokenizer.word_index
        X = self.tokenizer.texts_to_sequences(self.df['cut_review'].values)
        
        # padding X and let all the colonnes have the same length
        X = pad_sequences(X, maxlen = self.MAX_SEQUENCE_LENGTH)

        # multilabel in onehot coding
        Y = pd.get_dummies(self.df['cat_id']).values
        # split train and test with 10% test and 90% train
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, stratify = Y, test_size = 0.10, random_state = 42)
        # print(f'x shape 1 {X.shape[1]}')
        
        # define model
        self.lstm_model = Sequential() 
        self.lstm_model.add(Embedding(self.MAX_NB_WORDS, self.EMBEDDING_DIM, input_length=X.shape[1]))
        self.lstm_model.add(SpatialDropout1D(0.5))
        self.lstm_model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, activation='tanh'))
        self.lstm_model.add(Dense(10, activation='softmax'))
        self.lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(f'Model Summary : {self.lstm_model.summary()}')
        
        y_integers = np.argmax(Y_train, axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = dict(enumerate(class_weights))
        
        epochs = 6
        batch_size = 256
        start_time = time.time()
        history = self.lstm_model.fit(X_train, Y_train, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            validation_split=0.1,
                            class_weight=d_class_weights,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        
        print("--- %s seconds ---" % (time.time() - start_time))
        accr = self.lstm_model.evaluate(X_test,Y_test)
        print(f'Test set\n  Loss: {accr[0]}\tAccuracy: {accr[1]}')
        
        y_pred = self.lstm_model.predict(X_test)
        y_pred = y_pred.argmax(axis = 1)
        Y_test = Y_test.argmax(axis = 1)
        
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show();
        # print(history.history.keys())
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.show();

 
        # print(accuracy_score(y_pred, Y_test))
        print(classification_report(Y_test, y_pred,target_names=self.cat_id_df['cat'].values))
        
        if save_model:
            self.save()
    

    def predict_sentence(self,sentences):
        cats = {0:'书籍(livres)',
                1:'平板(tablettes)',
                2:'手机(téléphones portables)',
                3:'水果(fruits)',
                4:'洗发水(shampoing)',
                5:'热水器(chauffe-eau)',
                6:'蒙牛(Mengniu lait)',
                7:'衣服(vêtements)',
                8:'计算机(ordinateurs)',
                9:'酒店(hôtels)'}
        if isinstance(sentences,str):
            sentences = [sentences]
        for sen in sentences:
            sent = [" ".join([w for w in list(jieba.cut(remove_punctuation(sen))) if w not in stopwords])]
            # print(sen)
            seq = self.tokenizer.texts_to_sequences(sent)
            # print(seq)
            padded = pad_sequences(seq, maxlen=self.MAX_SEQUENCE_LENGTH)
            pred = self.lstm_model.predict(padded)
            cat_id= pred.argmax(axis=1)[0]
            # print(cat_id)
            # print(cat_id)
            print(f'{sen} : {cats[cat_id]}')
        # return cat_id_df[cat_id_df.cat_id==cat_id]['cat'].values[0]


if __name__ == '__main__':
    
    # create an instance of Model_LSTM
    lstm_model = Model_LSTM()
    
# =============================================================================
#     train a lstm model
# =============================================================================
    lstm_model.pipeline(save_model=False)
    
# =============================================================================
#     load a lstm model if don't want to train a new one
# =============================================================================
    # lstm_model.load_model('lstm_model.h5','lstm_tokenizer.m')
    
# =============================================================================
#    predict
#   牛奶好喝 : le lait est bon (reference : Mengniu lait)
#   用了几次发现头好痒，感觉过敏了: Après l'avoir utilisé plusieurs fois, j'ai trouvé que ma tête me démangeait et qu'elle était allergique(reference : Shampooing)
#   信号很差 : le signal est mauvais (reference : téléphone-portables)
#   手写识别还可以:La reconnaissance de l'écriture manuscrite est OK.(reference : téléphone-portables)
#   房间挺大，就是价格贵了点: La chambre est assez grande, mais le prix est un peu cher (reference : hôtel)
# =============================================================================
    # test_sentence = ["蒙牛好喝 ",'用了几次发现头好痒，感觉过敏了','信号很差','手写识别还可以','房间挺大，就是价格贵了点']
    # lstm_model.predict_sentence(test_sentence)

    
        
    
    
    
    
    
        