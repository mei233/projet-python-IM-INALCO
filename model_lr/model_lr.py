#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 17:13:36 2021

@author: meg
"""
'''
Ce script est de l'objectif d'entraîner le classifieurs de Logisitic regression avec différentes 
combinaisons des features sur les commentaiers chinois des plusieurs catégories
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
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
    

class ModelLR:
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
        self.clf2 = joblib.load(model_file) 
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
        joblib.dump(self.clf2, "LR_model.m")
        print('model saved as LR_model.m')
        joblib.dump(self.vectorizer, "tfidfvectorizer_model.m")
        print('model saved as tfidfvectorizer_model.m')
    
    def TfidfVector(self, ngram_range = (1, 2)):
        '''
        ngram_range = (1, 1) : unigramme
        Vous pouvez changez en (2,2) qui est bigramme, ou (1,2) qui est unigramme et bigramme, selon vos besoins
        
        create tfidf vector 

        Returns
        -------
        BTrain_X : class 'scipy.sparse.csr.csr_matrix'
            training data of tfidfvector
        BTest_X : class 'scipy.sparse.csr.csr_matrix'
            test data of tfidfvector
        BTrain_Y : class 'scipy.sparse.csr.csr_matrix'
            training label of tfidfvector
        BTest_Y : class 'scipy.sparse.csr.csr_matrix'
            test label of tfidfvector
        '''
        
        print("Treatment with tfidfVector\n")
        #transfer them according to different choices
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range,norm='l2',max_df = 0.8)
        VTrain_X = self.vectorizer.fit_transform(self.Train_X)
        
        VTest_X = self.vectorizer.transform(self.Test_X)
        print("training LR model...")
        self.train_catLR(VTrain_X,VTest_X)
        # self.draw_repartition()
        print("training Naive Bayes model...")
            
    
    def train_catLR(self,train_X,test_X):
        # C = [0.1,0.5,1,1.5,2,5]
        C = [5]
        res = []
        for c in C:
            self.clf2 = LogisticRegression(C=c,max_iter=1000,multi_class="ovr",solver='lbfgs')
            self.clf2.fit(train_X, self.Train_YC)
            predict = self.clf2.predict(test_X)
            score = self.clf2.score(test_X, self.Test_YC)
            print(classification_report(self.Test_YC,predict))
            print(f'c={c} : {score}')
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
        
        print("----- Mode : tf-idf, unigramme ----- ")
        start_time = time.time()
        self.TfidfVector(ngram_range = (1, 1))
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
        
        pred_result = self.clf2.predict(sentences)
        
        # pred_result = pred_result.argmax(axis=1)
        return [cats[y] for y in pred_result]
            

        

if __name__ == '__main__':
    
    train_model = ModelLR()
    
# =============================================================================
#     train a lstm model
#   save_model=True   --> save the trained model
# =============================================================================
    train_model.pipeline(save_model=False)
    
# =============================================================================
#     load a lstm model if don't want to train a new one
# =============================================================================
    # train_model.load_model('LR_model.m','tfidfvectorizer_model.m')
    
# =============================================================================
#    predict
#   牛奶好喝 : le lait est bon (reference : Mengniu lait)
#   用了几次发现头好痒，感觉过敏了: Après l'avoir utilisé plusieurs fois, j'ai trouvé que ma tête me démangeait et qu'elle était allergique(reference : Shampooing)
#   信号很差 : le signal est mauvais (reference : téléphone-portables)
#   手写识别还可以:La reconnaissance de l'écriture manuscrite est OK.(reference : téléphone-portables)
#   房间挺大，就是价格贵了点: La chambre est assez grande, mais le prix est un peu cher (reference : hôtel)
# =============================================================================
    # test_sentence = ["蒙牛好喝 ",'用了几次发现头好痒，感觉过敏了','信号很差','手写识别还可以','房间挺大，就是价格贵了点']
    # predict_label_list = train_model.predict_sentence(test_sentence)
    # print(f'Prediction result : {predict_label_list}')
    
    

