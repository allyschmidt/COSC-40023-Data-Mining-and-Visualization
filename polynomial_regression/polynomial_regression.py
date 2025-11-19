"""
Created on Wed Feb 26 00:13:59 2025

@author: allyschmidt
"""

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('Disease_Data.csv')
X = dataset.iloc[:, 0:-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()


# Finding the optimal degree using p-value
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
sc_X = StandardScaler()
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(len(y), 1)).flatten()


# Begin polynomial regression
poly_feature = PolynomialFeatures(degree = 2)
X_poly = poly_feature.fit_transform(X)
X_poly[:,1:] = sc_X.fit_transform(X_poly[:,1:])
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()

# There is no p-value above the significance level (0.05), so polynomial regression continues
poly_feature = PolynomialFeatures(degree = 3)
X_poly = poly_feature.fit_transform(X)
X_poly[:,1:] = sc_X.fit_transform(X_poly[:,1:])
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()

# There is no p-value above the significance level (0.05), so polynomial regression continues
poly_feature = PolynomialFeatures(degree = 4)
X_poly = poly_feature.fit_transform(X)
X_poly[:,1:] = sc_X.fit_transform(X_poly[:,1:])
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()

# There is no p-value above the significance level (0.05), so polynomial regression continues
poly_feature = PolynomialFeatures(degree = 5)
X_poly = poly_feature.fit_transform(X)
X_poly[:,1:] = sc_X.fit_transform(X_poly[:,1:])
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()

# There is no p-value above the significance level (0.05), so polynomial regression continues
'''
poly_feature = PolynomialFeatures(degree = 6)
X_poly = poly_feature.fit_transform(X)
X_poly[:,1:] = sc_X.fit_transform(X_poly[:,1:])
regressor = sm.OLS(endog = y, exog = X_poly).fit()
regressor.summary()
'''
# x1's p-value (0.993) is now above the significance level, so the previous degree (5) is optimal

# Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Day')
plt.ylabel('Cumulative Cases (Scaled)')
plt.show()

# Printing a single prediction
new_X_poly = poly_feature.transform([[365]])
new_X_poly[:,1:] = sc_X.transform(new_X_poly[:,1:])
print(regressor.predict(new_X_poly))
