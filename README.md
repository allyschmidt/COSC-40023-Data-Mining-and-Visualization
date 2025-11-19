# COSC-40023-Data-Mining-and-Visualization
This repository contains my work implementing various machine learning algorithms in both Python and R.

## Implementations

**Data Preprocessing** - Handles missing values using median and mean imputation, encodes categorical variables with one-hot encoding and label encoding, splits data into training and test sets, and applies feature scaling with normalization.

**Simple Linear Regression** - Predicts used car sale prices from list prices. Includes model training, coefficient extraction, visualizations for training and test sets, and predictions for new data points.

**Multiple Linear Regression** - Predicts crime rates based on factors like police funding, education levels, and demographics. Uses backward elimination with p-value testing at a 0.05 significance level to identify optimal features.

**Polynomial Regression** - Models disease spread over time by testing polynomial degrees from 2 through 6. Determines optimal degree using p-values (0.05 significance level) and generates trendline visualizations with observation data points.

**SVR and Random Forest Regression** - Compares Support Vector Regression with RBF kernel against Random Forest Regression (500 trees) on housing price data. Calculates adjusted R-squared over 10 runs to evaluate model performance.

**Logistic Regression** - Classifies customer purchase behavior using polynomial features (degree 2). Includes feature scaling, confusion matrix evaluation, and decision boundary visualizations on scaled features.

**Support Vector Machines** - Classifies iris species using SVM with multiple kernels tested (linear, polynomial degree 3, RBF, and sigmoid). Includes feature scaling and decision boundary visualization for the optimal kernel.

## Technologies Used

**Python:** scikit-learn, pandas, numpy, matplotlib, statsmodels

**R:** caTools, caret, ggplot2, e1071, randomForest
