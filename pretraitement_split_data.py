#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:22:22 2021

@author: mei
"""
'''
Ce script est pour l'objectif de prendre les prétraitements et la division du corpus en train et test pour les méthodes
de la Naive Bayes et la Regression Logistiques
'''
import re
import jieba
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
def remove_punctuation(line):
    '''
        save the model
        
        input : string

        Returns
        -------
        string without punctuations
    '''
    
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

def stopwordslist(file): 
    '''

    Parameters
    ----------
    file : file csv
        the file of chinese stopwords .

    Returns
    -------
    stopwords : a list of stopwords

    '''
    stopwords = []
    for line in open(file, 'r', encoding='utf-8').readlines():
        liste = line.strip().split(',')
    stopwords.extend(liste)  
    return stopwords  

stopwords = stopwordslist("chineseStopWords.txt")

def SupprimerDoublons(CSV_File):
    '''

    Parameters
    ----------
    CSV_File : a csv file 
        the corpus file.

    Returns
    -------
    data : 
        type pandas.

    '''
    data = pd.read_csv(CSV_File,sep=',',engine='python',error_bad_lines=False)
    # print(data.shape)
    data = data.drop_duplicates(subset=['review'], keep='first', inplace=False)
    # print(data.shape)
    return data

def segmentation(data):
    '''
    

    Parameters
    ----------
    data : pandas
        corpus.

    Returns
    -------
    data : corpus after segment with Jieba

    '''
    for row in data.index:
        ligne = data.loc[row,'review']
        # print(ligne)
        seg_list = jieba.cut(ligne,use_paddle=True)
        data.loc[row,'review'] = ' '.join(list(seg_list))
        # print(data.loc[row].values[2])
    return data


def pretraitement(file_in, train_file, test_file): 
    '''
    pretreatement if necessary

    Parameters
    ----------
    file_in : corpus csv
    train_file : 
        return file for training
    test_file : 
        return file for testing

    Returns
    -------
    None.

    '''
    # Step - a : supprimer les doublons
    Corpus = SupprimerDoublons(file_in)
    
    # Step - b : Remove blank rows if any.
    Corpus.dropna(axis=0, how='any', inplace=True)
    
    Corpus['cat'] = Corpus['cat'].factorize()[0]
    
    # Step - c : Remove the category that we don't need.
    # Corpus = Corpus[Corpus.cat!='蒙牛']
    
    #Step - c1 ： prétaitment-segmentation
    # Corpus = segmentation(Corpus)
    # Corpus['review'] = Corpus['review'].apply(remove_punctuation)
    
    #step - c3 : segmenation and/or delect stopwords, choose only one
    # Corpus['review'] = Corpus['review'].apply(lambda x: " ".join([w for w in list(jieba.cut(x))]))
    Corpus['review'] = Corpus['review'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
    
    # step - d: delect those commentaire which less than 5 segments
    Corpus['length'] = Corpus['review'].str.split().str.len()
    Corpus = Corpus[Corpus['length'] >= 8]
    
    #split en train / test
    # Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['tweet'],Corpus['existence'],test_size=0.3, random_state = 1)
    train, test = train_test_split(Corpus, stratify = Corpus.cat, random_state=42, test_size=0.2, shuffle=True)
    # stratify permet de la répartition homogène des categories pour les données de test
    
    print(train.shape)
    print(test.shape)
    
    #train to file csv
    train_f={
             'cat':train.cat.values,
        'review':train.review.values
        }
    data = DataFrame(train_f)
    data.to_csv(train_file,index=None,sep=",")
    
    #test to file csv
    test_f={
            'cat':test.cat.values,
        'review':test.review.values
        }
    data_test = DataFrame(test_f)
    data_test.to_csv(test_file,index=None,sep=",")
    

file_in = "corpus.csv"
train_file = "train_file.csv"
test_file = "test_file.csv"

pretraitement(file_in,train_file,test_file)