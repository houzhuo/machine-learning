#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:57:08 2018

@author: houzhuo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header = None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header = None)

X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_train[np.arange(64)]
y_test = digits_train[64]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)


#使用ARI进行聚类性能评估
from sklearn import metrics
print metrics.adjusted_rand_score(y_test,y_pred)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
plt.subplot(3,2,1)

x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])

X = np.array(zip(x1,x2)).reshape(len(x1),2)

plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instance')
plt.scatter(x1,x2)

colors = ['b','g','r','c','m','y','k','b']
markers = ['o','s','D','v','^','p','*','+']

clusters = [2,3,4,5,8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter+=1
    plt.subplot(3,2,subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    
    for i,l in enumerate(kmeans_model.labels_):#i is No,l is class
        plt.plot(x1[i],x2[i],color = colors[l],marker = markers[l],ls = 'None')
        plt.xlim([0,10])
        plt.ylim([0,10])
        print i,l
    sc_score = silhouette_score(X,kmeans_model.labels_,metric = 'euclidean')
    sc_scores.append(sc_score)
    print sc_scores
    plt.title('K=%s,silhouette coefficient = %0.03f'%(t,sc_score))
plt.figure()
plt.plot(clusters,sc_scores,'*-')
plt.xlabel('Number of CLusters')
plt.ylabel('Silhouette Coefficient Score')
plt.show()
        
        
#==========================elbow =====================================

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5,1.5,(2,10))
cluster2 = np.random.uniform(5.5,6.5,(2,10))
cluster3 = np.random.uniform(3.0,4.0,(2,10))
X = np.hstack((cluster1,cluster2,cluster3)).T
plt.scatter(X[:,0],X[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#测试九种不同聚类中心数量下，每种情况的聚类质量
K=range(1,10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis = 1))/X.shape[0])

plt.plot(K,meandistortions,'bx-')
plt.xlabel('k')
plt.ylabel('Ave Dispersion') 
plt.title('Select k with the Elbow Method')
plt.show()