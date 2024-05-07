# Libraries
library(glmnet)     #  This library contains the functions for training regularised linear regression models.
library(DescTools)  #  Descriptive statistics library.
library(ggplot2)

data <- read.csv("data.csv")

set.seed(2)  #  Setting random seed for random sample reproducibility

# Gathering training indices: 67% for training/cross-val, the rest for test.
train_indices <- sample(1:nrow(data), nrow(data) * 0.67, replace = FALSE)

# Now we'll partition the data as per the test design/plan.
train <- data[train_indices,]
test <- data[-train_indices,]

################## Data Understanding Code #####################
Str(data)     # Display the structure of the data
View(data)
summary(data)   # Summary statistics of the data
dim(data)       # Dimensions of the data (rows and columns)
nrow(data)      # Number of rows in the data
ncol(data)     # Number of columns in the data


## DescTools::
# We've used Desc() to gather descriptive statistics and visualisations of data before. 
Desc(data) 

# Calculate the mean for each variable
mean_values <- sapply(data, mean, na.rm = TRUE)
print(mean_values)

# Calculate the median for each variable
median_values <- sapply(data, median, na.rm = TRUE)
print(median_values)

# Function to calculate mode
calculate_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Calculate the mode for each variable
mode_values <- sapply(data, calculate_mode)
print(mode_values)

# Calculate the range for each variable
range_values <- sapply(data, function(x) max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
print(range_values)

# Calculate the variance for each variable
variance_values <- sapply(data, var, na.rm = TRUE)
print(variance_values)

# Calculate the standard deviation for each variable
std_dev_values <- sapply(data, sd, na.rm = TRUE)
print(std_dev_values)

# Count missing values for each variable
missing_values <- colSums(is.na(data))
print(missing_values)

# Identify rows with missing values
rows_with_missing <- which(rowSums(is.na(data)) > 0)
print(rows_with_missing)

# View the row with missing values
print(data[369, ])

# Remove the row with missing values
data <- data[-369, ]

### Correlation###
# Exploring associations using Pearson's correlation statistic 
# is important if we might wish to explore the data using
# linear regression. cor(data) below facilitates calculations 
# of correlation matrices based on input matrices or data
# frames.
train.corr <- cor(train)
print(train.corr)

ggcorrplot::ggcorrplot(train.corr)  #  Correlation heat map

# Faceted scatter plots of Total Floor Area (V.2) and Preliminary Estimated Construction Cost (V.4) by Project Locality (V.1)
ggplot(data, aes(x = V.2, y = V.4)) + 
  geom_point() +
  facet_wrap(~V.1) +
  labs(title = "Scatter Plots of V.2 vs V.4 by Project Locality", x = "Total Floor Area (m^2)", y = "Preliminary Estimated Construction Cost (IRR)")

# Correlation matrix
correlation_matrix <- cor(data[, c("V.1","V.2", "V.3", "V.4", "V.5", "V.6", "V.7", "V.8", "Y")], use = "complete.obs")
print(correlation_matrix)

# Identify potential multicollinearity (absolute correlation > 0.7)
col_to_check <- which(abs(correlation_matrix) > 0.7 & correlation_matrix != 1, arr.ind = TRUE)
if (any(col_to_check)) {
  print(paste0("High correlation detected between: ", rownames(correlation_matrix)[col_to_check[, 1]], " and ", 
               rownames(correlation_matrix)[col_to_check[, 2]]))
  # Further investigation needed to decide if variable removal or other techniques are necessary
} else {
  print("No high correlations (> 0.7) detected among relevant variables.")
}

#################### Data Preparation Code ######################

library(tidyverse)  #  Streamlined data manipulation, visualization and analysis
library(caret)      #  Building, training, evaluating and tuning models

head(train, 4)
tail(train, 4)
class(train) # show the data type
str(train)  # Checking the structure/format of the data.
summary(train)

train <- mutate_at(train, vars(V.1, V.5, V.7, V.8, Y), as.numeric) # convert total to numeric variable
str(train)                                 # let`s check the data structure again


is.na(train)                                     # classic way to check NA`s
sum(is.na(train))                                # counting NA`s
apply(is.na(train),2, which)                     # which indexes of NA`s (df only)
which(complete.cases(train))                     # identify observed complete values

train <- na.omit(train)

clean.vector <- na.omit(list(train))               # clean/remove a vector NA`s
clean.df <- na.omit(train)                         # clean/remove a dataframe NA`s
apply(is.na(clean.df),2, which)                   # make sure if there are missing values 

any(is.na(clean.vector))

any(is.na(clean.df))

train %>% pull() %>% head()                 # extract column values of `state` as a vector
print(train)

#Histogram of a numerical variable
hist(train$V.2, 
     main = "Histogram of V.2 (Total Floor Area)", 
     xlab = "Total Floor Area (m^2)", ylab = "Frequency", 
     col = "skyblue", border = "black")
abline(v = mean(train$V.2), 
       col = "red", 
       lwd = 2) # Add a vertical line for the mean
legend("topright", 
       legend = paste("Mean:", 
                      round(mean(train$V.2), 2)), 
       col = "red", 
       lwd = 2) # Add a legend for the mean

#Boxplot of a numerical variable to identify outliers
boxplot(train$V.4, train$V.2,
        main = "Boxplot of Preliminary Estimated Construction Cost (IRR)",
        xlab = "Variable V.4",
        ylab = "Construction Cost (IRR)",
        col = "skyblue",     # Customize colors
        border = "black",
        notch = TRUE,        # Add a notch
        pch = 19,            # Adjust outlier symbol
        horizontal = FALSE, # Horizontal orientation
        grid = TRUE         # Add grid lines
)
boxplot(train$V.2,
        main = "Boxplot of Preliminary Estimated Construction Cost (IRR)",
        xlab = "Variable V.4",
        ylab = "Construction Cost (IRR)",
        col = "skyblue",     # Customize colors
        border = "black",
        notch = TRUE,        # Add a notch
        pch = 19,            # Adjust outlier symbol
        horizontal = FALSE, # Horizontal orientation
        grid = TRUE         # Add grid lines
)
# scatter plot
plot(train$V.2, train$V.4, 
     main = "Scatter Plot: Total Floor Area vs Preliminary Estimated Construction Cost",
     xlab = "Total Floor Area (m^2)",
     ylab = "Preliminary Estimated Construction Cost (IRR)",
     col = "blue",
     pch = 19)

#bar plot
barplot(table(train$V.1), 
        main = "Bar Plot of Project Locality",
        xlab = "Project Locality",
        ylab = "Frequency",
        col = "skyblue")

# Time series plot of Actual Construction Costs (Y) over time (V.7)
plot(train$V.7, train$Y,
     type = "l",
     main = "Line Plot: Duration of Construction vs Actual Construction Costs",
     xlab = "Duration of Construction (Months)",
     ylab = "Actual Construction Costs (IRR)",
     col = "red")

# heatmap
heatmap(cor(train[, c("V.1","V.2", "V.3", "V.4", "V.5", "V.6", "V.7", "V.8", "Y")]),
        main = "Correlation Heatmap",
        xlab = "Variables",
        ylab = "Variables")

# Example density plot
plot(density(train$V.8),
     main = "Density Plot of Price per Unit Area",
     xlab = "Price per Unit Area (IRR)",
     col = "blue")

pairs(train[, c("V.2", "V.3", "V.4", "V.5", "V.6", "V.7", "V.8", "Y")])

heatmap(cor(train[, c("V.2", "V.3", "V.4", "Y")]), 
        main = "Correlation Heatmap (Subset of Variables)", 
        xlab = "Variables", 
        ylab = "Variables")

#### Handling of the outliers########
#by examining each variable by boxplot, I have found outliers in 2 variables i.e. V.2 and V.4

# Boxplot before removing outliers
par(mfrow=c(1, 2))
boxplot(data$V.2, main="Boxplot of V.2 (Before)")
boxplot(data$V.4, main="Boxplot of V.4 (Before)")

# Identify outliers for V.2 and V.4 variables
outliers_V2 <- boxplot.stats(data$V.2)$out
outliers_V4 <- boxplot.stats(data$V.4)$out

# Remove outliers from the dataset
cleaned_data <- data[!(data$V.2 %in% outliers_V2 | data$V.4 %in% outliers_V4), ]

# Boxplot after removing outliers
par(mfrow=c(1, 2))
boxplot(cleaned_data$V.2, main="Boxplot of V.2 (After)")
boxplot(cleaned_data$V.4, main="Boxplot of V.4 (After)")

# Summary of removed outliers
cat("Outliers removed from V.2:", outliers_V2, "\n")
cat("Outliers removed from V.4:", outliers_V4, "\n")

# Remove rows with NA values
cleaned_data <- na.omit(cleaned_data)

# Summary of cleaned dataset
cat("Summary of cleaned dataset:\n")
summary(cleaned_data)

### IQR Method - to identify & handle outliers ###

# The IQR is used to identify and deal with outliers. It is a measure of 
# the spread of the data values. It is a reliable measure of 
# dispersion because it is not affected by extreme values of outliers.
#
# In the IQR method a range is defined by using first and third quartile and a 
# multiplier which is usually set as 1.5. All the values below the lower  
# limit and above the upper limit are considered as outliers.

# Calculate IQR for each variable
Q1 <- apply(train, 2, quantile, probs = 0.25)
Q3 <- apply(train, 2, quantile, probs = 0.75)
IQR <- Q3 - Q1

outliers <- apply(train, 2, function(x) x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR))

cleaned_data <- train
cleaned_data[outliers] <- NA
cleaned_data <- na.omit(cleaned_data)

summary(cleaned_data)

##################  Modelling Code ###################

# Create Predictor matrix & Response vector for both train & test set
x_train <- as.matrix(train[ , -1])   # predictor matrix for the training set.
y_train <- train[ , 1]            # response vector for the training set.

x_test <- as.matrix(test[ , -1])   # predictor matrix for the test set.
y_test <- test[ , 1]              # response vector for the test set.

# Excluding variables V.1 and V.3 from the model and adding predictor matrix
# in the training & test set 

x_train_excl_V3 <- as.matrix(x_train[ , -3])
View(x_train_excl_V3)      # checking the correct variable has been removed
x_test_excl_V3 <- x_test[ , -3]
View(x_test_excl_V3)      # checking the correct variable has been removed

# a matrix is a two-dimensionsal collection of elements of the same data type 
# (numeric, character, or logical) arranged into a fixed number of rows and columns. 

## Standardizing the predictor variables ##

# Standardizing the predictor variables to have mean 
# zero and unit variance. This is important for ridge regression 
# because it ensures that the penalty term is applied equally to 
# all the coefficients. Using scale function to do this.

x_train <- scale(x_train)  # standardize the training predictors
x_test <- scale(x_test)    # standardize the test predictors

x_train_excl_V3 <- scale(x_train_excl_V3)


##### Build Training Model - Cross Validation using Ridge Regression ####

# To perform ridge regression, I’ll use functions from the glmnet package. 

# I’ll use the glmnet() function to fit the ridge regression model 
# and specify alpha=0 to select Ridge Regression

# Setting alpha equal to 1 is equivalent to using Lasso Regression and 
# setting alpha to some value between 0 and 1 is equivalent to using an elastic net.

# I’ll use the default values of alpha and lambda and 
# let the function choose the optimal values for us. 

# Alpha = 0 corresponds to ridge regression, 
# Alpha = 1 corresponds to lasso regression and 
# 0 < alpha < 1 corresponds to elastic net regression, 
# a combination of ridge and lasso.

# Ideally, producing multiple models and based on the lowest mean squared 
# error and decide on the final model for implementation.

## Fitting the ridge regression model ##

# Model 1 for ridge regression
ridge_model <- glmnet(x_train, y_train, alpha = 0, standardize = FALSE) 

# Model 2 for Lasso regression
lasso_model <- glmnet(x_train, y_train, alpha = 1, standardize = FALSE)  

# Model 3 removing variable 3
ridge_model_excl_V3 <- glmnet(x_train_excl_V3, y_train, alpha = 0, standardize = FALSE)   

# (The predictor value/ input features will need to be amended to exclude V3 in this example)

# Model 4 - Ridge Regression with lambda_min
ridge_model_lambda_min <- glmnet(x_train, y_train, alpha=0, lambda = ridge_model$lambda.min, standardize = FALSE)

# Model 5 Lasso regression with lambda_min
lasso_model_lambda.min <- glmnet(x_train, y_train, alpha = 1, lambda = lasso_model$lambda.min, standardize = FALSE) 

# Note that by default, the glmnet() function standardizes the
# variables so that they are on the same scale. To turn off this default setting,
# use the argument standardize=FALSE.

# View summary model which will show  the length, class, mode, and dimensions of the elements.
summary(ridge_model)
summary(ridge_model_lambda_min)
summary(lasso_model_lambda.min)

# The corresponding values of lambda, beta, df, dev. ratio, and a0. 
# The beta element is a sparse matrix, which means that it only stores 
# the non-zero values of the coefficients. 
# The df element is the degrees of freedom, the number of non-zero coefficients. 
# The dev.ratio element is the fraction of deviance the model explains. 
# The a0 element is the intercept term.

# Associated with each value of  λ is a vector of ridge regression coefficients, 
# stored in a matrix that can be accessed by coef()
# coef(model)

# To check the coefficients using the dim()
dim(coef(ridge_model))

ridge_model

# Associated with each value of λ is a vector of ridge regression coefficients,
# stored in a matrix that can be accessed by coef(). In this case, it is a 
# 5X100 matrix, with 5 rows (one for each predictor, plus an intercept) and
# 100 columns (one for each value of lambda).

ridge_model$lambda[20]

coef(ridge_model) [ , 20]

# Plotting the model object using plot function, shows coef change as a func of lambda

plot(ridge_model, xvar = "lambda", label = TRUE)# plot the coefficients vs lambda

# The x-axis is on a log scale (Log Lambda), so the smaller lambda 
# values are on the right, and the larger values are on the left. 

# The y-axis shows the values of the coefficients, and each line 
# corresponds to a different predictor variable.

# In this example, when lambda log is 8, the coefficients are essentially zero.
# When we relax lambda the coefficients grow away from zero in a smooth way.
# The sum of squares of the coefficients are getting bigger and bigger until 
# we reach a point where Lambda is effectively zero & the coefficients
# are regularized & so these would be the coefficients  that you get from an 
# ordinary least squares fit of these variables. 

### Perform k-fold cross-validation to find optimal lambda value ###

# Next, we’ll identify the lambda value that produces the 
# lowest test mean squared error (MSE) by using k-fold 
# cross-validation a technique that splits the data into 
# several subsets and uses some for training and some for testing.

# glmnet has the function cv.glmnet() that automatically 
# performs k-fold cross validation. A fold is a subset of the 
# data used for testing, while the rest is used for training. 
# The default value is 10, meaning the data is split into 10 subsets, 
# and each subset is used as a test set once. 

# Model 1 - Ridge Regression cross validation
cv_ridge_model <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 5) 
# Performs 5 fold cross validation on ridge model

# Model 2 - Lasso Regression cross validation
cv_lasso_model <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5) 
# Performs 5 fold cross validation on lasso model

# Model 3 - Ridge Regression with variable 3 excluded 
cv_ridge_model_excl_V3 <- cv.glmnet(x_train_excl_V3, y_train, alpha = 1, nfolds = 5) 

# Model 4 - Ridge Regression with lambda.min
# ridge_model_lambda_min <- glmnet(x_train, y_train, alpha=0, lambda = cv_ridge_model$lambda.min)
cv_ridge_model_lambda_min <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5) 

# Model 5 - Lasso Regression with lambda.min
cv_lasso_model_lambda_min <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5) 

summary(cv_ridge_model)
summary(ridge_model_lambda_min)

# The cvm element is the mean cross-validated error for each value of lambda. 

# The cvsd element is the standard deviation of the cross-validated error for 
# each lambda value. 

# The cvup and cvlo elements are the upper and lower confidence bounds 
# for the cross-validated error for each lambda value. 

# The nzero element is the number of non-zero coefficients for each value of lambda. 

# The lambda.min element is the value of lambda that gives the minimum
# cross-validated error. 

# The lambda.1se element is the largest value of lambda, giving a 
# cross-validated error within one standard error of the minimum.

## plotting the cv_model object using the plot function ##

# Shows the cross-validated error changes as a function of lambda. 

# The x-axis is on a log scale, so the smaller lambda values are on 
# the right, and the larger values are on the left. 

# The y-axis shows the values of the cross-validated error, and the 
# error bars show the confidence bounds. 

# The vertical dotted lines indicate the values of lambda that give 
# the minimum cross-validated error and the largest error within 
# one standard error of the minimum.

plot(cv_ridge_model) # plot the cross-validated error vs lambda


# This is a plot of the cross validated MSE and from the right hand side
# it dips downs. In the beginning the MSE is very high and the coefficients 
# are restricted to be too small and then it starts to level off. 
# This indicates that the full model is doing a good job.

# There are two vertical lines:
# The first one indicates the min. MSE.
# The second indicates the one standard error of the min. MSE.
# This is a more restricted model that can do as well as the min. MSE and
# we can decide to use this value instead of the min. MSE.

plot(cv_lasso_model)  # Model 2 Lasso Regression
plot(cv_ridge_model_excl_V3)  # Model 3 Ridge Regression with V3 removed
plot(cv_ridge_model_lambda_min) # Model 4 Ridge regression with lambda min.
plot(cv_lasso_model_lambda_min) # Model 5 Lasso regression with lambda min.

### This will show the results of the best results from each model. ###

cv_ridge_model
cv_lasso_model
cv_ridge_model_excl_V3
cv_ridge_model_lambda_min
cv_lasso_model_lambda_min

# This will pick the coefficient corresponding to the best model.  
coef(cv_ridge_model)
coef(cv_lasso_model_lambda_min)
coef(ridge_model_lambda_min)

#### Let's store the validation results! They will be useful to compare against the test set results.
# Model 1 Ridge model MSE & RMSE
cross_validation_ridge_MSE <- min(cv_ridge_model$cvm) 
cross_validation_ridge_RMSE <- sqrt(cross_validation_ridge_MSE)  

# Model 2 Lasso model MSE & RMSE
cross_validation_lasso_MSE <- min(cv_lasso_model$cvm)  
cross_validation_lasso_RMSE <- sqrt(cross_validation_lasso_MSE)  

# Model 3 Ridge model excluding Var. 3 MSE & RMSE
cross_validation_ridge_excl_V3_MSE <- min(cv_ridge_model_excl_V3$cvm) 
cross_validation_ridge_excl_V3_RMSE <- sqrt(cross_validation_ridge_excl_V3_MSE)    

# Model 4 Ridge model with lambda min. MSE & RMSE  
cross_validation_ridge_lambda_min <- min(cv_ridge_model_lambda_min$cvm) 
cross_validation_ridge_lambda_min_RMSE <- sqrt(cross_validation_ridge_lambda_min)    # The square root of the MSE of the ridge regression model excl V3

# Model 5 Lasso model with lambda min.
cross_validation_lasso_lambda_min <- min(cv_lasso_model_lambda_min$cvm) 
cross_validation_lasso_lambda_min_RMSE <- sqrt(cross_validation_lasso_lambda_min)    # The square root of the MSE of the ridge regression model excl V3

# Create a data frame to store the MSE and RMSE values
model_comparison <- data.frame(
  Model = c("Ridge", "Lasso", "Ridge Excl. V3", "Ridge Lambda Min", "Lasso Lambda Min"),
  MSE = c(cross_validation_ridge_MSE, cross_validation_lasso_MSE, cross_validation_ridge_excl_V3_MSE, cross_validation_ridge_lambda_min, cross_validation_lasso_lambda_min),
  RMSE = c(cross_validation_ridge_RMSE, cross_validation_lasso_RMSE, cross_validation_ridge_excl_V3_RMSE, cross_validation_ridge_lambda_min_RMSE, cross_validation_lasso_lambda_min_RMSE)
)

# Print the data frame
print(model_comparison)

#  min() can be used to get the smallest MSE from the evaluated lambda values.
#  The square root of the MSE of the ridge regression model

### Find optimal lambda value that minimizes test MSE ###

# From the above results we can see the Ridge Lamda Min has produced the lowest MSE. 
#         Model         MSE               RMSE
# 1            Ridge  30.13858         5.489862
# 2            Lasso  22.94890         4.790501
# 3   Ridge Excl. V3  29.86710         5.465080
# 4 Ridge Lambda Min  20.59304         4.537955
# 5 Lasso Lambda Min  20.91619         4.573422

# I've selected the optimal value of lambda, it's time for me 
# to assess the performance of the ridge regression model on the test set.

# I'll utilize the predict function to generate predictions of the response variable f
# or the test set, employing the ridge regression model with the chosen lambda value. 

# Next, I'll compare these predicted values with the actual values and compute 
# various metrics to gauge the accuracy of the predictions, 
# including Mean Squared Error (MSE), 
# Root Mean Squared Error (RMSE), and the 
# coefficient of determination (R-squared).
# Using the predict function, 

# I'll generate the predicted values for the test set using prediction func;

### To determine the optimal lambda value that minimizes cross-validated error, element.
# Lower values of lambda indicate stronger regularization, 
# while higher values indicate weaker regularization
cv_ridge_model_lambda_min$lambda.min 

####################  Final Evaluation ####################

test <- na.omit(test)   # Remove NA value from test set
anyNA(test)  #  If there are any missing values in the test set they will need to be handled.

# The test set predictor matrix has already been created earlier.
x_test <- as.matrix(test[ , -1])   # predictor matrix for the test set.
y_test <- test[ , 1]
x_test <- scale(x_test)


is.infinite(x_test)
is.infinite(y_test)

head(x_test)
head(y_test)

# Check if the test set is scaled similarly to the training set
summary(x_test)

# Make test set predictions.
predictions <- predict(ridge_model_lambda_min, s = ridge_model_lambda_min$lambda, newx = x_test)

summary(predictions)    # Verify the prediction process

# Finally, let's evaluate how well (or poorly) we have done on the test set.

test_MSE <- MSE(predictions, y_test)
test_RMSE <- RMSE(predictions, y_test)

test_MSE 
test_RMSE

# Extracting the best lambda by indexing the glmnet lambda component
# then index it by order of RMSE, order puts them in ascending order
# smallest value and this will pick out the best lambda.

lam.best <- ridge_model_lambda_min$lambda[order(test_MSE)[1]]

lam.best <- min(ridge_model_lambda_min$lambda)
lam.best

coef(ridge_model_lambda_min, s = lam.best)

############## Plot of test set predictions vs actual values ######################

# Pad predictions vector with zeros if its length is less than y_test
if (length(predictions) < length(y_test)) {
  extra_zeros <- rep(0, length(y_test) - length(predictions))
  predictions <- c(predictions, extra_zeros)
} else if (length(predictions) > length(y_test)) {
  predictions <- predictions[1:length(y_test)]
}

# Check if the lengths are the same
length(predictions)
length(y_test)

# If they are not the same, you can adjust one of the vectors to match the length of the other.
# For example, you can trim or pad one of the vectors to match the length of the other.
# But in this case its same

# Ensuring both vectors have the same length
min_length <- min(length(predictions), length(y_test))
predictions <- predictions[1:min_length]
y_test <- y_test[1:min_length]

# Now plot the data
plot(x = predictions, y = y_test, frame = FALSE, pch = 19, 
     col = "red", xlab = "Predicted Values", ylab = "Actual Values")

######Dual line chart for predicted vs. actual values########
test_instances <- seq_along(y_test)

plot(x = test_instances, y = y_test, frame = FALSE, pch = 19, type = "l",
     col = "red", xlab = "Test Instance", ylab = "Valence")

lines(x = test_instances, y = predictions, pch = 18, col = "blue", type = "l", lty = 2)  

# Adding legend
legend("topleft", legend=c("Actual", "Predicted"), col=c("red", "blue"), lty = 1:2, cex=0.8) 


###### verify if the model predictions fall within  +/- 500,000 Iranian Rial##############

# Calculate the absolute difference between predictions and actual values
abs_diff <- abs(predictions - y_test)

# Check if the absolute difference is within the specified threshold
within_threshold <- abs_diff <= 500000

# Count the number of predictions within the threshold
num_within_threshold <- sum(within_threshold)

# Calculate the percentage of predictions within the threshold
percentage_within_threshold <- (num_within_threshold / length(y_test)) * 100

print(percentage_within_threshold)

# Print the results
cat("Number of predictions within +/- 500,000 Iranian Rial threshold:", num_within_threshold, "\n")
cat("Percentage of predictions within +/- 500,000 Iranian Rial threshold:", percentage_within_threshold, "%\n")

#####ploting the visual for it
# Convert the logical vector to a factor with labels "Within Threshold" and "Outside Threshold"
within_threshold_factor <- factor(within_threshold, levels = c(FALSE, TRUE), labels = c("Outside Threshold", "Within Threshold"))

# Create a bar plot to visualize the proportion of predictions within the threshold
bar_colors <- c("red", "green")
bar_names <- c("Outside Threshold", "Within Threshold")
barplot(table(within_threshold_factor), col = bar_colors, main = "Predictions Within +/- 500,000 Iranian Rial threshold", ylab = "Frequency")

# Add a legend to the plot
legend("topleft", legend = bar_names, fill = bar_colors, cex = 0.8)

#########Plotting another scatter plot####################

# Calculate the mean or median of the actual values
actual_mean <- mean(y_test)
actual_median <- median(y_test)

# Calculate the threshold values
threshold_upper <- actual_mean + 500000
threshold_lower <- actual_mean - 500000

# Create a vector to indicate whether each prediction is within the threshold
within_threshold <- ifelse(predictions >= threshold_lower & predictions <= threshold_upper, TRUE, FALSE)

# Define colors for points based on whether they are within the threshold
point_colors <- ifelse(within_threshold, "blue", "red")

# Create the scatter plot
plot(predictions, y_test, col = point_colors, pch = 19, 
     xlab = "Predicted Values", ylab = "Actual Values", 
     main = "Scatter Plot with Threshold Highlighted")

# Add horizontal lines for the threshold range
abline(h = c(threshold_upper, threshold_lower), col = "green", lty = 2)

# Add legend
legend("topright", legend = c("Within Threshold", "Outside Threshold"), 
       col = c("blue", "red"), lwd = 3, cex = 0.5, text.width = 0.9)


