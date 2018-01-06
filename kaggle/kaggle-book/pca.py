#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:54:14 2018

@author: houzhuo
"""

import numpy as np
M = np.array([[1,2],[2,4]])
np.linalg.matrix_rank(M,tol = None)#返回矩阵的秩

import pandas as pd
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header = None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header = None)

X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

from sklearn.decomposition import PCA
#将64维压缩至2维
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)

from matplotlib import pyplot as plt

def plot_pca_scatter():
    colors = ['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
    for i in xrange(len(colors)):
        px = X_pca[:,0][y_digits.as_matrix()==i]#等于类别i的x，y坐标
        py = X_pca[:,1][y_digits.as_matrix()==i]
        plt.scatter(px,py,c = colors[i])
    
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
    
plot_pca_scatter()

#在相同配置的svm上进行学习
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

from sklearn.svm import LinearSVC

svc = LinearSVC()
svc.fit(X_train,y_train)
y_predict = svc.predict(X_test)


# 使用PCA将原64维的图像数据压缩到20个维度。
estimator = PCA(n_components=20)

# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征。
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转化（transform）。
pca_X_test = estimator.transform(X_test)#???????????????????????为什么只要transform

# 使用默认配置初始化LinearSVC，对压缩过后的20维特征的训练数据进行建模，并在测试数据上做出预测，存储在pca_y_predict中。
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_predict = pca_svc.predict(pca_X_test)

# 从sklearn.metrics导入classification_report用于更加细致的分类性能分析。
from sklearn.metrics import classification_report

# 对使用原始图像高维像素特征训练的支持向量机分类器的性能作出评估。
print svc.score(X_test, y_test)
print classification_report(y_test, y_predict, target_names=np.arange(10).astype(str))

# 对使用PCA压缩重建的低维图像特征训练的支持向量机分类器的性能作出评估。
print pca_svc.score(pca_X_test, y_test)
print classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str))

    