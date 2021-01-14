#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:13:36 2021

@author: meg
"""
'''
Ce script est de l'objectif d'entraîner le classifieurs de Naive Bayes avec différentes 
combinaisons des features sur les commentaiers chinois des plusieurs catégories
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn import naive_bayes
# from sklearn.externals import joblib
import joblib
import time
import jieba
le = LabelEncoder()


class predict_pretaitement(object):
    '''
    For predict the sentence 
    '''
    def __init__(self):
        
        self.stopwords = []
        self.stopwordslist("chineseStopWords.txt")
        
    
    def stopwordslist(self,filepath): 
        for line in open(filepath, 'r', encoding='utf-8').readlines():
            liste = line.strip().split(',')
        self.stopwords.extend(liste)  
    
    def delectStopwords(self,sen):
        sen = " ".join([w for w in list(jieba.cut(sen)) if w not in self.stopwords])
        return sen
    

class ModelNB:
    '''
    build a logistic regression model
    '''
    def __init__(self):
        self.train_file = "train_fileD.csv"
        self.test_file = "test_fileD.csv"
        
    def load_model(self,model_file,vector_file):
        '''
    
        Parameters
        ----------
        model_file : 
            the path and the file name of the model file
        vector_file : TYPE
            the path and the file name of the tfidfvector model file

        Returns
        -------
        None.

        '''
        print(f"Loading model {model_file}...")
        self.vectorizer = joblib.load(vector_file)
        print(f"Loading model {vector_file}...")
        self.naive= joblib.load(model_file) 
        # self.model = joblib.load(load_model_file) 
            
    def Graphes(self,x,y):
        '''
        Print out the learning curve of the model
        
        Returns
        -------
        None.

        '''
        print("Print out the graph...\n")
        ax1 = plt.subplot()
        ax1.set_xscale('log')
        ax1.plot(x,y,marker='o',color='g')
        ax1.set_xlabel('c')
        ax1.set_ylabel('Accuracy LR')
        ax1.grid()
        plt.show()
        
    def save(self):
        '''
        save the model

        Returns
        -------
        None.

        '''
        print('Saving model...\n')
        joblib.dump(self.naive, "NB_model.m")
        print('model saved as NB_model.m')
        joblib.dump(self.vectorizer, "countvectorizer_model.m")
        print('model saved as countvectorizer_model.m')
    
    def simpleBow(self,ngram_range=(1,1)):
        
        '''
        input ngram_range

        Returns
        -------
        None.
        '''
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)
        BTrain_X = self.vectorizer.fit_transform(self.Train_X)
        BTest_X = self.vectorizer.transform(self.Test_X)
        print("training NB model...")
        # self.train_polariteLR(BTrain_X,BTest_X)
        self.train_catNB(BTrain_X,BTest_X)
        
        print("training Naive Bayes model...")
            
    
    def train_catNB(self,train_X,test_X):
        # C = [0.1,0.5,1,1.5,2,5]
        C = [0.1]
        res = []
        for c in C:
            self.naive = naive_bayes.MultinomialNB(alpha=c)
            self.naive.fit(train_X, self.Train_YC)
            predict = self.naive.predict(test_X)
            # accuracy = accuracy_score(predict, self.Test_YC)*100
            # print(f"Accuracy Score -> {accuracy}")
            score = self.naive.score(test_X, self.Test_YC)
            print(classification_report(self.Test_YC,predict))
            print(f'a={c} : {score}')
            res.append(score)
        # self.Graphes(C,res)
    
    
    def pipeline(self,save_model=False):
        '''
        main function : training the model out of the train file and test file using logistic regression 

        Parameters
        ----------
        save_model : optional
             The default is False. 
             If it's Ture, it will save a model
        plot : optional
            The default is False.  
            If it's Ture, it will plot the graph

        Returns
        -------
        None.

        '''
        print("Loading training data...")
        self.Train_corpus = pd.read_csv(self.train_file, sep=',',engine='python',error_bad_lines=False)
        self.Train_corpus.dropna(axis=0, how='any', inplace=True)
        print("Loading test data...")
        self.Test_corpus = pd.read_csv(self.test_file, sep=',',engine='python',error_bad_lines=False)
        self.Test_corpus.dropna(axis=0, how='any', inplace=True)
        
        
        self.Train_X = self.Train_corpus['review']
        self.Test_X = self.Test_corpus['review']
        self.Train_YC = le.fit_transform(self.Train_corpus['cat'].values).astype('float')
        self.Test_YC = le.fit_transform(self.Test_corpus['cat']).astype('float')
        
        print("----- Mode : BOW, unigramme ----- ")
        start_time = time.time()
        self.simpleBow(ngram_range = (1, 1))
        print("--- %s seconds ---" % (time.time() - start_time))
        
        if save_model:
            self.save()
        
    def predict_sentence(self,sentences):
        '''
        predict a category for a list of sentences

        Parameters
        ----------
        sentences : a list of str
            input a list of sentences for predict

        Returns
        -------
        list
            the prediction cat by the model

        '''
        predict = predict_pretaitement()
        
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
            
        sentences = self.vectorizer.transform([predict.delectStopwords(entry) for entry in sentences])
        # print(sentences)
        
        pred_result = self.naive.predict(sentences)
        
        # pred_result = pred_result.argmax(axis=1)
        return [cats[y] for y in pred_result]
            

        

if __name__ == '__main__':
    
    train_model = ModelNB()
    
# =============================================================================
#     train a naive bayes model
#   save_model=True   --> save the trained model
# =============================================================================
    train_model.pipeline(save_model=False)
    
# =============================================================================
#     load a naive bayes model if don't want to train a new one
# =============================================================================
    # train_model.load_model('NB_model.m','countvectorizer_model.m')
    
# =============================================================================
#     predict
#   牛奶好喝 : le lait est bon (reference : Mengniu lait)
#   用了几次发现头好痒，感觉过敏了: Après l'avoir utilisé plusieurs fois, j'ai trouvé que ma tête me démangeait et qu'elle était allergique(reference : Shampooing)
#   信号很差 : le signal est mauvais (reference : téléphone-portables)
#   手写识别还可以:La reconnaissance de l'écriture manuscrite est OK.(reference : téléphone-portables)
#   房间挺大，就是价格贵了点: La chambre est assez grande, mais le prix est un peu cher (reference : hôtel)
# =============================================================================
    # test_sentence = ["蒙牛好喝 ",'用了几次发现头好痒，感觉过敏了','信号很差','手写识别还可以','房间挺大，就是价格贵了点']
    # predict_label_list = train_model.predict_sentence(test_sentence)
    # print(f'Prediction result : {predict_label_list}')
    
    

