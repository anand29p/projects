# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 06:51:21 2021

@author: Anand
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
dataset = pd.read_csv("Data_Pre.csv")
dataset

# Creating the matrix of features. In the imported dataset Country, Age, Salary are independent variables
X = dataset.iloc[:, :-1].values
X
# Creating the dependent variable
y = dataset.iloc[:, 3].values
y
...
# Let’s see the missing values in the data set.
dataset.isna().sum()

# Taking care of missing data
from sklearn.impute import SimpleImputer 

# Creating the object of Imputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# fit imputer object to data X (Matrix of feature X)
imputer = imputer.fit(X[:, 1:3]) 

# Replace the missing data of column by mean
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

# Encoding categorical data
# Creating dummy variables using OneHotEncoder class
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

# Creating the object of LabelEncoder class
labelencoder_X = LabelEncoder()

# fit labelencoder_X object to first coulmn Country of matrix X
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X

# Normalization (Min-Max Scalar)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(X[:, :])
data_scaled

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split

# Choosing 20% data as test data, so we will have 80% data in training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
