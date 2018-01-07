#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 10:43:53 2018

@author: houzhuo
"""

import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
y = titanic['survived']
X = titanic.drop(['row.names','name','survived'],axis = 1)

X['age'].fillna(X['age'].mean(),inplace = True)
X.fillna('UNKOWN',inplace = True)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 33)

#类别型特征向量化
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))
print len(vec.feature_names_)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy')#以信息熵为标准作为划分
dt.fit(X_train,y_train)
dt.score(X_test,y_test)

#筛选前20％个特征
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile = 20)
X_train_fs =fs.fit_transform(X_train,y_train)#???????????????为什么两个
dt.fit(X_train_fs,y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs,y_test)

#通过交叉验证，按照固定间隔的百分比筛选特征，并做图
from sklearn.cross_validation import cross_val_score
import numpy as np
percentiles = range(1,100,2)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile = i)
    X_train_fs =fs.fit_transform(X_train,y_train)
    scores = cross_val_score(dt,X_train,y_train,cv=5)
    results = np.append(results,scores.mean())
print results

#寻找最优特征数
opt = np.where(results == results.max())[0]
print 'Optimal number of features %d' %percentiles[opt]


#画图
import pylab as pl
pl.plot(percentiles,results)

pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

    
# 使用最佳筛选后的特征，利用相同配置的模型在测试集上进行性能评估。
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
dt.score(X_test_fs, y_test)
    
    
    