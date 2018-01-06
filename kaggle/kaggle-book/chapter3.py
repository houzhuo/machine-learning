#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:53:24 2018

@author: houzhuo
"""
#对字典存储的数据进行特征抽取和向量化
measurement = [{'city':'Dubai','temperature':33.},{'city':'London','temperature':12.},{'city':'San Fransisco','temperature':18.}]
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print vec.fit_transform(measurement).toarray()
print vec.get_feature_names()

#使用DictVectorrizer 并且不去掉停用词，对文本特征进行向量化的朴素贝叶斯分类性能测试
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size = 0.25,random_state = 33)

from sklearn.feature_extraction.text import CountVectorizer#引入替换成TfidfVectorizer
count_vec = CountVectorizer()
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)




from sklearn.naive_bayes import MultinomialNB
mnb_count = MultinomialNB()
mnb_count.fit(X_count_train,y_train)
print 'accuracy of classify using NB (countVectorizer without filtering stopword)',mnb_count.score(X_count_test,y_test)
y_count_predict = mnb_count.predict(X_count_test)

from sklearn.metrics import classification_report
print classification_report(y_test,y_count_predict,target_names= news.target_names)

#去掉停用词
count_filter_vec = CountVectorizer(analyzer = 'word',stop_words= 'english')
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train,y_train)
mnb_count_filter.score(X_count_filter_test,y_test)

