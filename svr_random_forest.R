
# Importing the dataset
dataset = read.csv('Housing_Data.csv')

# SVR

# Split the dataset into the Training set and Test set
library(caTools)
split = sample.split(dataset$Y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting SVR to the training set
library(e1071)
regressor = svm(formula = Y ~ .,
                data = training_set,
                type = 'eps-regression',
                kernel = 'radial')

# Making a prediction
y_pred = predict(regressor, newdata = test_set)

# Calculate adjusted R-squared
ssr = sum((test_set$Y - y_pred) ^ 2)
sst = sum((test_set$Y - mean(test_set$Y)) ^ 2)
r2 = 1 - (ssr/sst)
print(r2)
r2_adjusted = 1 - (1 - r2) * (length(test_set$Y) - 1) / (length(test_set$Y) - 5)
print(r2_adjusted)

# 10 calculated adjusted R-squared values:
# 0.6618986, 0.7480653, 0.654488, 0.5343588, 0.7605925, 0.8022176, 0.6795678, 0.5222025, 0.6355827, 0.6383548
# Average adjusted R-squared (from the 10 values): 0.6637329


# Random Forest Regression

# Split the dataset into the Training set and Test set
library(caTools)
split = sample.split(dataset$Y, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Regression to the training set
library(randomForest)
regressor = randomForest(formula = Y ~ .,
                         data = training_set,
                         ntree = 500)

# Making a prediction
y_pred = predict(regressor, newdata = test_set)

# Calculate adjusted R-squared
ssr = sum((test_set$Y - y_pred) ^ 2)
sst = sum((test_set$Y - mean(test_set$Y)) ^ 2)
r2 = 1 - (ssr/sst)
print(r2)
r2_adjusted = 1 - (1 - r2) * (length(test_set$Y) - 1) / (length(test_set$Y) - 5)
print(r2_adjusted)      

# 10 calculated adjusted R-squared values:
# 0.723385, 0.7602306, 0.5581353, 0.76973, 0.7678142, 0.8303739, 0.5411711, 0.7800704, 0.8132085, 0.5687729
# Average adjusted R-squared (from the 10 values): 0.71128919

# The Random Forest Regression model had a better performance, with an average adjusted r-squared 
# value of 0.71128919, compared to 0.6637329 with SVR


