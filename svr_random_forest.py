# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Housing_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()


#SVR

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)

y_train_scaled = sc_y.fit_transform(y_train.reshape(len(y_train), 1)).flatten()
y_test_scaled = sc_y.transform(y_test.reshape(len(y_test), 1)).flatten()

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train_scaled, y_train_scaled)

# Making a prediction
y_pred_scaled = regressor.predict(X_test_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(len(y_pred_scaled), 1))

# Calculate adjusted R-squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)
r2_adjusted = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 5)
print(r2_adjusted)

# 10 calculated adjusted R-squared values:
# 0.713146, 0.713634, 0.588939, 0.675036, 0.708971, 0.711422, 0.738083, 0.566091, 0.749758, 0.542278
# Average adjusted R-squared (from the 10 values): 0.670736


# Random Forest Regression

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Training the Random Forest Regression model on the Training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500)
regressor.fit(X_train, y_train)

# Making a prediction
y_pred = regressor.predict(X_test)

# Calculate adjusted R-squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)
r2_adjusted = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 5)
print(r2_adjusted)

# 10 calculated adjusted R-squared values:
# 0.763731, 0.661303, 0.764938, 0.549310, 0.654420, 0.458737, 0.710402, 0.709855, 0.720449, 0.583867
# Average adjusted R-squared (from the 10 values): 0.6577012


# The SVR model had a better performance, with an average adjusted r-squared value of 0.670736, 
# compared to 0.6577012 with Random Forest Regression



