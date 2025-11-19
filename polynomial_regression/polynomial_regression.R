# Import dataset
dataset = read.csv('Disease_Data.csv')

# Fitting Polynomial Regression to the whole dataset
dataset$Day2 = dataset$Day^2
regressor = lm(formula = Cumulative.Cases ~ .,
               data = dataset)

# Finding the optimal degree using p-value
dataset$Day3 = dataset$Day^3
regressor = lm(formula = Cumulative.Cases ~ .,
               data = dataset)
summary(regressor)

# There is no p-value above the significance level (0.05), so polynomial regression continues
dataset$Day4 = dataset$Day^4
regressor = lm(formula = Cumulative.Cases ~ .,
               data = dataset)
summary(regressor)

# There is no p-value above the significance level (0.05), so polynomial regression continues
dataset$Day5 = dataset$Day^5
regressor = lm(formula = Cumulative.Cases ~ .,
               data = dataset)
summary(regressor)

# There is no p-value above the significance level (0.05), so polynomial regression continues
# dataset$Day6 = dataset$Day^6
# regressor = lm(formula = Cumulative.Cases ~ .,
#                data = dataset)
# summary(regressor)
# Day's p-value (0.992620) is now above the significance level, so the previous degree (5) is optimal

# Visualizing the Polynomial Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Day, y = dataset$Cumulative.Cases),
             colour = 'red') +
  geom_line(aes(x = dataset$Day, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Polynomial Regression') +
  xlab('Day') +
  ylab('Cumulative.Cases')

# Printing a single prediction
newdata = data.frame(Day = 365,
                      Day2 = 365^2,
                      Day3 = 365^3,
                      Day4 = 365^4,
                      Day5 = 365^5
                     )

print(predict(regressor, newdata))
