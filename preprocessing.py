"""
Created on Wed Jan 22 20:19:05 2025

@author: allyschmidt
"""

# Import libraries
import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Customer_Data.csv')
X = dataset.iloc[:,:-1].to_numpy()
y = dataset.iloc[:,-1].to_numpy()

# Handle missing data in Age column
from sklearn.impute import SimpleImputer
ages_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X[:,1:2] = ages_imputer.fit_transform(X[:,1:2])

# Handle missing data in Salary column
salary_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:,2:3] = salary_imputer.fit_transform(X[:,2:3])

# Encode the independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],
                                       remainder='passthrough')
X = ct.fit_transform(X)

# Encode the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature Scaling using normalization
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
X_train[:,3:] = mm.fit_transform(X_train[:,3:])
X_test[:,3:] = mm.transform(X_test[:,3:])

