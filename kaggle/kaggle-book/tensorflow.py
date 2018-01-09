#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:50:33 2018

@author: houzhuo
"""
from sklearn import datasets,metrics,preprocessing,cross_validation
import numpy as np
boston = datasets.load_boston()
X,y = boston.data,boston.target
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.25,random_state = 33)

#数据标准化
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



import tensorflow as tf
greeting=tf.constant('Hello Google Tensorflow')
sess = tf.Session()
result = sess.run(greeting)
print result
sess.close()

#===============================linear classifier with tensorflow==============
import tensorflow as tf
import numpy as np
import pandas as pd

train = pd.read_csv('/Users/houzhuo/Downloads/breast-cancer-train.csv')
test = pd.read_csv('/Users/houzhuo/Downloads/breast-cancer-test.csv')

X_train = np.float32(train[['Clump Thickness','Cell Size']].T)
y_train = np.float32(train[['Type']].T)
X_test = np.float32(test[['Clump Thickness','Cell Size']].T)
y_test = np.float32(test[['Type']].T)
#定义tensorflow类型变量b作为线性模型的截距，初始值为1
b= tf.Variable(tf.zeros([1]))
#w作为系数，初始值为－1，1之间均匀分布的随机数
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))

y=tf.matmul(W,X_train)+b
#使用tensorflow里的reduce mean取得训练集上的均方误差
loss = tf.reduce_mean(tf.square(y-y_train))
#使用梯度下降估计参数w，b，并且设置步长0.01
optimizer = tf.train.GradientDescentOptimizer(0.01)
#以最小二乘损失loss为优化目标
train = optimizer.minimize(loss)
#初始化所有变量
init = tf.initialize_all_variables()
sess = tf.Session()
#执行变量初始化操作
sess.run(init)
#迭代1000次
for step in xrange(0,1000):
    sess.run(train)
    if step %200 ==0:
        print step,sess.run(W),sess.run(b)


#测试并且绘图
test_negative = test.loc[test['Type']==0][['Clump Thickness','Cell Size']]
test_postive = test.loc[test['Type']==1][['Clump Thickness','Cell Size']]

import matplotlib.pyplot as plt
plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(test_postive['Clump Thickness'],test_postive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
lx = np.arange(0,12)
ly = (0.5-sess.run(b)-lx * sess.run(W)[0][0])/sess.run(W)[0][1]
plt.plot(lx,ly,color='green')
plt.show()