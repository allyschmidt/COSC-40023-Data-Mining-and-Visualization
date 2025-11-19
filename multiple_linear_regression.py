"""
Created on Tue Feb 11 14:38:01 2025

@author: allyschmidt
"""

# Import libraries
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Crime_Data.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,
                                                    random_state = 0)

# Building the optimal model using Backward Elimination
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train.values.reshape(len(y_train), 1)).flatten()


# Backward Elimination
import statsmodels.api as sm
X_train = sm.add_constant(X_train)

X_opt = X_train[:,[0, 1, 2, 3, 4, 5, 6]]
regressor_opt = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_opt.summary()

# x3 has the largest P-value (0.945), so that column is removed from X_opt and backward elimination is continued
# x3 corresponds to X3

X_opt = X_train[:,[0, 1, 2, 4, 5, 6]]
regressor_opt = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_opt.summary()

# x4 has the largest P-value (0.232), so that column is removed from X_opt and backward elimination is continued
# x4 corresponds to X5

X_opt = X_train[:,[0, 1, 2, 4, 6]]
regressor_opt = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_opt.summary()

# x4 has the largest P-value (0.100), so that column is removed from X_opt and backward elimination is continued
# x3 corresponds to X6

X_opt = X_train[:,[0, 1, 2, 4]]
regressor_opt = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_opt.summary()

# All P-values are now under the significance level (0.05), so backward elimination is complete and the optimal model is found.
# Best team of independent variables: X1, X2, and X4


X_test = sc_X.transform(X_test)
X_test = sm.add_constant(X_test)
X_test_opt = X_test[:, [0, 1, 2, 4]]
y_pred = regressor_opt.predict(X_test_opt)

print(regressor_opt.params)

# Prediction for X1=500, X2=50, X4=30
new_X = sc_X.transform([[500, 50, 40, 30, 20, 10]])
new_X = sm.add_constant(new_X, has_constant='add')
new_y_pred = regressor_opt.predict(new_X[:,[0, 1, 2, 4]])
print(new_y_pred)

















