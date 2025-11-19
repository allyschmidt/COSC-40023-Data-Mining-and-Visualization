# Import dataset
dataset = read.csv('Crime_Data.csv')

# Split the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Y, SplitRatio = 0.9)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fit Multiple Linear Regression to the Training set
regressor = lm(formula = Y ~ .,
               data = training_set)

# Predict the Test set results
y_pred = predict(regressor, newdata = test_set)

#above two methods were not needed

#Build the optimal model using Backward Elimination
regressor_opt = lm(formula = Y ~ X1 + X2 + X3 + X4 + X5 + X6,
                   data = training_set)
summary(regressor_opt)

# X3 has the largest P-value (0.799423), so that column is eliminated and backward elimination is continued

regressor_opt = lm(formula = Y ~ X1 + X2 + X4 + X5 + X6,
                   data = training_set)
summary(regressor_opt)

# X6 has the largest P-value (0.245700), so that column is eliminated and backward elimination is continued

regressor_opt = lm(formula = Y ~ X1 + X2 + X4 + X5,
                   data = training_set)
summary(regressor_opt)

# X5 has the largest P-value (0.408954), so that column is eliminated and backward elimination is continued

regressor_opt = lm(formula = Y ~ X1 + X2 + X4,
                   data = training_set)
summary(regressor_opt)

# All P-values are now under the significance level (0.05), so backward elimination is complete and the optimal model is found.
# Best team of independent variables: X1, X2, and X4

y_pred = predict(regressor_opt, newdata = test_set)
# solution says y_pred = predict(regressor_opt, newdata = test_set[, c('X1', 'X2', 'X4')])

print(regressor_opt$coefficients)
print(predict(regressor_opt, newdata = data.frame('X1'=c(500),'X2'=c(50),'X4'=c(30))))

