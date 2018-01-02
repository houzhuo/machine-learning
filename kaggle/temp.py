# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd



df_train = pd.read_csv('/Users/houzhuo/Downloads/breast-cancer-train.csv')

df_test = pd.read_csv('/Users/houzhuo/Downloads/breast-cancer-test.csv')


df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness','Cell Size']]
df_test_postive = df_test.loc[df_test['Type'] == 1][['Clump Thickness','Cell Size']]

import matplotlib.pyplot as plt

plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker ='o',s = 200,c='red')
plt.scatter(df_test_postive['Clump Thickness'],df_test_postive['Cell Size'],marker ='x',s = 150,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()


intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0,12)
ly = (-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c='yellow')


#==============================================================================
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][:10])
print 'Testing accuracy(10 examples):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type'])




intercept = lr.intercept_
coef = lr.coef_[0,:]
#lx*coef[0] + ly*coef[1]+intercept = 0
ly = (-intercept-lx*coef[0])/coef[1]

plt.plot(lx,ly,c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker ='o',s = 200,c='red')
plt.scatter(df_test_postive['Clump Thickness'],df_test_postive['Cell Size'],marker ='x',s = 150,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.show()