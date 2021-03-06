import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("train.csv")  # loading dataset
#  dataset = dataset.apply(pd.to_numeric, errors='coerce')
dataset = pd.get_dummies(dataset)
dataset.fillna(0, inplace=True)

X = np.array(dataset.values[:, 2:377]) #  data features
Y = np.array(dataset.values[:, 1]) #  data label (what we predict)

X = preprocessing.scale(X)
dataset.dropna(inplace=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)

print (accuracy)