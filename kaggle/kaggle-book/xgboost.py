#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:07:41 2018

@author: houzhuo
"""

import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

X=titanic[['pclass','age','sex']]
y=titanic['survived']

X['age'].fillna(X['age'].mean(),inplace = True)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))
print len(vec.feature_names_)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
print 'The accuracy is :',rfc.score(X_test,y_test)

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train,y_train)
print 'The xgboost score:',xgbc.score(X_test,y_test)