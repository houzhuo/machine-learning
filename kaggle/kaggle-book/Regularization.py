#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:35:06 2018

@author: houzhuo
"""

X_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

import numpy as np
#在x轴上从0，25均匀采集100个数据点
xx = np.linspace(0,26,100)
#变成一行
xx = xx.reshape(xx.shape[0],1)
#用x轴的随机数据作为测试集，正好可以画条直线
yy = regressor.predict(xx)

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label = "Degree = 1")
plt.axis =([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Prize of Pizza')
plt.legend(handles = [plt1])
plt.show()
print 'score:',regressor.score(X_train,y_train)#????难道回归用训练集？




#使用二次多项式进行拟合
from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree =2)
#使用PolynomialFeatures(degree =2)映射出2次多项式特征
X_train_poly2 = poly2.fit_transform(X_train)
#尽管特征维度有提升，但是模型的基础仍是线性模型！
regressor2 = LinearRegression()
regressor2.fit(X_train_poly2,y_train)
#重新映射x轴数据
xx_poly2 = poly2.transform(xx)  #这里又变成了transform
yy_poly2 = regressor2.predict(xx_poly2)

plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label = "Degree = 1")
plt2,=plt.plot(xx,yy_poly2,label = "Degree = 2")#注意这里是xx！！！
plt.axis =([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Prize of Pizza')
plt.legend(handles = [plt1,plt2])
plt.show()
print 'score2:',regressor2.score(X_train_poly2,y_train)#????难道回归用训练集？

#再试试4项多项式
from sklearn.preprocessing import PolynomialFeatures
poly4 = PolynomialFeatures(degree =4)
#使用PolynomialFeatures(degree =4)映射出4次多项式特征
X_train_poly4 = poly4.fit_transform(X_train)
#尽管特征维度有提升，但是模型的基础仍是线性模型！
regressor4 = LinearRegression()
regressor4.fit(X_train_poly4,y_train)
#重新映射x轴数据
xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor4.predict(xx_poly4)

plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label = "Degree = 1")
plt2,=plt.plot(xx,yy_poly2,label = "Degree = 2")#注意这里是xx！！！
plt4,=plt.plot(xx,yy_poly4,label = "Degree = 4")
plt.axis =([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Prize of Pizza')
plt.legend(handles = [plt1,plt2,plt4])
plt.show()
print 'score4:',regressor4.score(X_train_poly4,y_train)


#＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝评估在测试集伤的表现
# 准备测试数据。
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]



print 'poly1',regressor.score(X_test,y_test)

X_test_poly2 = poly2.transform(X_test)
print 'poly2',regressor2.score(X_test_poly2,y_test)

X_test_poly4 = poly4.transform(X_test)
print 'poly3',regressor4.score(X_test_poly4,y_test)

#这是L1正则
#Lasso模型在4次多项式特征上的拟合表现,加入了惩罚项，让更多参数接近于0，
from sklearn.linear_model import Lasso
lasso_poly4 = Lasso()
lasso_poly4.fit(X_train_poly4,y_train)

print lasso_poly4.score(X_test_poly4,y_test)
print lasso_poly4.coef_

#普通的
print regressor4.score(X_test_poly4, y_test)

print regressor4.coef_



#L2正则（Ridge） 目的是减少参数之间的差异性！

#验证参数之间的巨大差异性
print np.sum(regressor4.coef_ ** 2)


from sklearn.linear_model import Ridge
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4,y_train)

print ridge_poly4.score(X_test_poly4,y_test)
print ridge_poly4.coef_
print np.sum(ridge_poly4.coef_ ** 2)

