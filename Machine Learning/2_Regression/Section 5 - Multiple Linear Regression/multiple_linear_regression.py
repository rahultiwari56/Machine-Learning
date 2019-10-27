#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
data = pd.read_csv("50_Startups.csv")

#splitting independent column into x & dependent column into y
x = data.iloc[:, :-1].values
y = data.iloc[:, 4].values


#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#removing 1st column to avoid confusion
x = x[:, 1:]

#splitting data into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#fitting multiple linear regression into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting test set result
y_pred = regressor.predict(x_test)

#model accuracy
from sklearn.metrics import r2_score
accuracy = r2_score(y_pred, y_test)

print("Model accuracy:",accuracy)