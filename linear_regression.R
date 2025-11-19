# Import the dataset
dataset = read.csv('Dealership_Data.csv')

# Split the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Sell.Price, SplitRatio = 3/4)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fit Simple Linear Regression to the Training set
regressor = lm(formula = Sell.Price ~ List.Price,
               data = training_set)

# Predict the Test set results
y_pred = predict(regressor, newdata = test_set)

# Visualize the Training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$List.Price, y = training_set$Sell.Price),
             colour = 'red') +
  geom_line(aes(x = training_set$List.Price, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('List Price vs Sale Price (Training set)') +
  xlab('List Price') +
  ylab('Sell Price')

# Visualize the Test set results
ggplot() +
  geom_point(aes(x = test_set$List.Price, y = test_set$Sell.Price),
             colour = 'red') +
  geom_line(aes(x = training_set$List.Price, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('List Price vs Sale Price (Test set)') +
  xlab('List Price') +
  ylab('Sell Price')

# Print slope and intercept of the model
print(regressor$coefficients)

# Print a prediction
print(predict(regressor, newdata = data.frame('List.Price' = c(20))))
