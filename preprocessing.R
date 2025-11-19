#Importing the dataset
dataset = read.csv('Customer_Data.csv')

#Handle missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     median(dataset$Age, na.rm = TRUE),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        mean(dataset$Salary, na.rm = TRUE),
                        dataset$Salary)

#Encode categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('India', 'Sri lanka', 'China'),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0,1))

# Split the dataset into training and test sets
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling using normalization
library(caret)
process = preProcess(training_set[,2:3], method = c("range"))
training_set[,2:3] = predict(process, training_set[,2:3])
test_set[,2:3] = predict(process, test_set[,2:3])


