#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:30:20 2018

@author: houzhuo
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


column_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size', 'Bare Nuclei',' Bland Chromatin','Normal Nucleoli','Mitoses','Class']

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names = column_names)
#将？替换为标准缺失值表示
data = data.replace(to_replace='?',value = np.nan)
#丢弃带有缺失值的数据
data = data.dropna(how = 'any')
data.shape

#标准化数据，保证每个纬度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值而主导
X_train,X_test,y_train,y_test = train_test_split(data[column_names[1:10]],data[column_names[10]],test_size = 0.25,random_state = 33)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

#分别用lr和SGD对样本进行预测
lr = LogisticRegression()
sgdc = SGDClassifier()
lr.fit(X_train,y_train)
lr_y_predict = lr.predict(X_test)

sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)


print 'accuracy of LR Classifier:', lr.score(X_test,y_test)#准确率
print classification_report(y_test,lr_y_predict,target_names=['Bengin','Malignant'])#其余三个指标
print classification_report(y_test,sgdc_y_predict,target_names=['Bengin','Malignant'])





#====================================手写体数据识别SVM=========================
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size = 0.25,random_state = 33)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

lsvc = LinearSVC()
lsvc.fit(X_train,y_train,)
y_predict = lsvc.predict(X_test)
print 'accuracy of SVC Classifier:', lsvc.score(X_test,y_test)#准确率

from sklearn.metrics import classification_report
print classification_report(y_test,y_predict,target_names = digits.target_names.astype(str))#其余三个指标


#====================================决策树====================================
import pandas as pd
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic.head()
titanic.info()
X = titanic[['pclass','age','sex']]
y = titanic['survived']
X.info()

#补齐缺失数据
X['age'].fillna(X['age'].mean(),inplace = True)
X.info()

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 33)
#特征抽取
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
#特征转换，类别型特征单独剥离出来，独成一列。数值型保持不变
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
print vec.feature_names_
X_test = vec.fit_transform(X_test.to_dict(orient = 'record'))
#['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male']

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict = dtc.predict(X_test)
from sklearn.metrics import classification_report
print dtc.score(X_test,y_test)
print classification_report(y_test,y_predict,target_names = ['died','survived'])#其余三个指标

#使用随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
#使用梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_predict = gbc.predict(X_test)

print classification_report(y_test,rfc_predict)#其余三个指标
print classification_report(y_test,gbc_predict,target_names = ['died','survived'])#其余三个指标