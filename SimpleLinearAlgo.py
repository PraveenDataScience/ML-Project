# -*- coding: utf-8 -*-
"""
Editor: PraveenDataScience

Simple Linear Algo:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the data set
dataset = pd.read_csv('Salary_Data.csv')

# Spliting data set into x & y:
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,1].values

# Dividing the dataset based on training & test

# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

# Implement our classifier based on Simple linear Regression

from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(X_train,Y_train)

y_predict=simplelinearRegression.predict(X_test)

# Implement the graph

plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,simplelinearRegression.predict(X_test))
plt.show()