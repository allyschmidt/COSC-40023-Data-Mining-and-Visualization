"""
Created on Fri Feb  7 19:14:31 2025

@author: allyschmidt
"""

# Import the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Dealership_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4,
                                                    random_state = 0)

# Train the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the Test set results
y_pred = regressor.predict(X_test)

# Visualize the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('List Price vs Sale Price (Training set)')
plt.xlabel('List Price')
plt.ylabel('Sale Price')
plt.show()

# Visualize the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('List Price vs Sale Price (Test set)')
plt.xlabel('List Price')
plt.ylabel('Sale Price')
plt.show()

# Print slope and intercept of the model
print('Slope:',regressor.coef_)
print('Intercept:', regressor.intercept_)

# Print a prediction
print('Predicted sale price of an item listed at $20k:', regressor.predict([[20]]))

