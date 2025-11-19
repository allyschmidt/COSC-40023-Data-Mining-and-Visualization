# Importing the dataset
dataset = read.csv('Iris_Data.csv')

# Encoding categorical data
dataset$Species = as.factor(dataset$Species)

# Feature Scaling
scaled_cols = scale(dataset[, 1:2])
dataset[, 1:2] = scaled_cols

# Fitting SVM model
library(e1071)
classifier = svm(formula = Species ~ .,
                 data = dataset,
                 type = 'C-classification',
                 kernel = 'radial')
                 
# Prediction
y_pred = predict(classifier, newdata = dataset)

# Showing the Confusion Matrix and Accuracy
library(caret)
cm = confusionMatrix(y_pred, dataset$Species)
print(cm$table)
print(cm$overall['Accuracy'])

# Visualizing the results
set = dataset
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Sepal.Length', 'Sepal.Width')
y_grid = predict(classifier, newdata = grid_set)
plot(NULL,
     main = 'SVM',
     xlab = 'Sepal Length (Scaled)', ylab = 'Sepal Width (Scaled)',
     xlim = range(X1), ylim = range(X2))
points(grid_set, pch = 20, col = c("#e63946", "#f1fa8c", "#06d6a0")[y_grid])
points(set, pch = 21, bg = c("#e6394699", "#f1fa8c99", "#06d6a099")[dataset$Species])
