# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')

# Encoding categorical data
dataset$Purchased = as.factor(dataset$Purchased)

# Add columns
dataset$Age2 = dataset$Age^2
dataset$Salary2 = dataset$EstimatedSalary^2
dataset$AgeSalary = dataset$Age * dataset$EstimatedSalary

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
training_scaled_cols = scale(training_set[,c(1, 2, 4, 5, 6)])
training_set[, c(1, 2, 4, 5, 6)] = training_scaled_cols
center = attr(training_scaled_cols, 'scaled:center')
scale = attr(training_scaled_cols, 'scaled:scale')
test_set[, c(1, 2, 4, 5, 6)] = scale(test_set[,c(1, 2, 4, 5, 6)],
                        center,
                        scale)

# Fitting Logistic Regression to the Training set
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set)
y_pred = as.factor(ifelse(prob_pred > 0.5, 1, 0))

# Showing the Confusion Matrix and Accuracy
library(caret)
cm = confusionMatrix(y_pred, test_set$Purchased)
print(cm$table)
print(cm$overall['Accuracy'])


# Visualizing the Training set results
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)

colnames(grid_set) = c('Age', 'EstimatedSalary')

grid_set$Age2 = grid_set$Age^2
grid_set$Salary2 = grid_set$EstimatedSalary^2
grid_set$AgeSalary = grid_set$Age * grid_set$EstimatedSalary

grid_set[, c(1, 2, 3, 4, 5)] <- sweep(grid_set[, c(1, 2, 3, 4, 5)], 2, scale, "*")
grid_set[, c(1, 2, 3, 4, 5)] <- sweep(grid_set[, c(1, 2, 3, 4, 5)], 2, center, "+")

grid_unscaled = grid_set[, c("Age", "EstimatedSalary")]

grid_unscaled$Age2 = grid_unscaled$Age^2
grid_unscaled$Salary2 = grid_unscaled$EstimatedSalary^2
grid_unscaled$AgeSalary = grid_unscaled$Age * grid_unscaled$EstimatedSalary

X_grid = as.data.frame(scale(grid_unscaled,center,scale))

prob_set = predict(classifier, type = 'response', newdata = X_grid)
y_grid = as.factor(ifelse(prob_set > 0.5, 1, 0))
plot(NULL, 
     main = 'Logistic Regression (Training set)',
     xlab = 'Age (Scaled)', ylab = 'Estimated Salary (Scaled)',
     xlim = range(X1), ylim = range(X2))
points(X_grid, pch = 20, col = c('tomato', 'springgreen3')[y_grid])
points(set, pch = 21, bg = c('red3', 'green4')[set$Purchased])

# Visualizing the Test set results
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')

grid_set$Age2 = grid_set$Age^2
grid_set$Salary2 = grid_set$EstimatedSalary^2
grid_set$AgeSalary = grid_set$Age * grid_set$EstimatedSalary

grid_set[, c(1, 2, 3, 4, 5)] <- sweep(grid_set[, c(1, 2, 3, 4, 5)], 2, scale, "*")
grid_set[, c(1, 2, 3, 4, 5)] <- sweep(grid_set[, c(1, 2, 3, 4, 5)], 2, center, "+")

grid_unscaled = grid_set[, c("Age", "EstimatedSalary")]

grid_unscaled$Age2 = grid_unscaled$Age^2
grid_unscaled$Salary2 = grid_unscaled$EstimatedSalary^2
grid_unscaled$AgeSalary = grid_unscaled$Age * grid_unscaled$EstimatedSalary

X_grid = as.data.frame(scale(grid_unscaled,center,scale))

prob_set = predict(classifier, type = 'response', newdata = X_grid)
y_grid = as.factor(ifelse(prob_set > 0.5, 1, 0))

plot(NULL,
     main = 'Logistic Regression (Test set)',
     xlab = 'Age (Scaled)', ylab = 'Estimated Salary (Scaled)',
     xlim = range(X1), ylim = range(X2))
points(X_grid, pch = 20, col = c('tomato', 'springgreen3')[y_grid])
points(set, pch = 21, bg = c('red3', 'green4')[set$Purchased])
