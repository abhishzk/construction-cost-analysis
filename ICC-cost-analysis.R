# Libraries
library(glmnet)     #  Regularised linear regression models (ridge / lasso).
library(DescTools)  #  Descriptive statistics library.
library(ggplot2)

# Business target: predictions must be within +/- 500,000 IRR.
# Y is recorded in units of 10,000 IRR (Data Descriptions), so the tolerance
# in the units of Y is 500000 / 10000 = 50.
TOLERANCE <- 500000 / 10000

data <- read.csv("data.csv")

################## Data Understanding Code #####################
Str(data)       # Structure of the data: 9 columns, all numeric/integer
summary(data)   # Summary statistics of the data
dim(data)       # Dimensions of the data (rows and columns)
nrow(data)      # Number of rows in the data
ncol(data)      # Number of columns in the data

# Count missing values for each variable
missing_values <- colSums(is.na(data))
print(missing_values)

# Identify rows with missing values (found programmatically, not hard coded)
rows_with_missing <- which(rowSums(is.na(data)) > 0)
print(rows_with_missing)
print(data[rows_with_missing, ])

# The only incomplete row is completely empty (a blank row at the end of the
# file), so it carries no information and is removed BEFORE partitioning.
data <- data[-rows_with_missing, ]
nrow(data)

# V.1 is a project locality (zip code), i.e. a category stored as a number.
# Its numeric size has no meaning, so it is never treated as a quantity.
data$V.1 <- factor(data$V.1)
nlevels(data$V.1)

## DescTools::
Desc(data)

# Descriptive statistics for the numeric variables
num_vars <- setdiff(names(data), "V.1")
num_data <- data[, num_vars]

calculate_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

descriptives <- data.frame(
  mean     = sapply(num_data, mean),
  median   = sapply(num_data, median),
  mode     = sapply(num_data, calculate_mode),
  range    = sapply(num_data, function(x) max(x) - min(x)),
  variance = sapply(num_data, var),
  sd       = sapply(num_data, sd)
)
print(round(descriptives, 2))

################## Data Partition ###################################
# Partition once, on the cleaned data, and set the test set aside.
# Everything that is learned from data (scaling, lambda, outlier limits)
# is learned from the training set only, so the test set is not leaked into.

set.seed(2)  #  Setting random seed for random sample reproducibility

# 67% for training/cross-validation, the rest for the final test.
train_indices <- sample(1:nrow(data), round(nrow(data) * 0.67), replace = FALSE)
train <- data[train_indices, ]
test  <- data[-train_indices, ]
nrow(train)
nrow(test)

################## Exploratory Analysis (training set only) #########

### Correlation ###
# Pearson's correlation is only meaningful for numeric variables, so the
# locality category V.1 is excluded.
train.corr <- cor(train[, num_vars])
print(round(train.corr, 3))

ggcorrplot::ggcorrplot(train.corr, lab = TRUE, lab_size = 3)  #  Correlation heat map

# Predictors strongly related to the target
print(round(sort(train.corr[, "Y"], decreasing = TRUE), 3))

# Potential multicollinearity (absolute correlation > 0.7 between predictors)
pred_corr <- train.corr[setdiff(num_vars, "Y"), setdiff(num_vars, "Y")]
high <- which(abs(pred_corr) > 0.7 & upper.tri(pred_corr), arr.ind = TRUE)
if (nrow(high) > 0) {
  print(paste0("High correlation between: ", rownames(pred_corr)[high[, 1]],
               " and ", colnames(pred_corr)[high[, 2]],
               " (r = ", round(pred_corr[high], 2), ")"))
  # Ridge/lasso regularisation is used because it copes with correlated predictors.
} else {
  print("No high correlations (> 0.7) detected among predictors.")
}

# Faceted scatter plots of Total Floor Area (V.2) and Preliminary Estimated
# Construction Cost (V.4) by Project Locality (V.1)
ggplot(train, aes(x = V.2, y = V.4)) +
  geom_point() +
  facet_wrap(~V.1) +
  labs(title = "Scatter Plots of V.2 vs V.4 by Project Locality",
       x = "Total Floor Area (m^2)",
       y = "Preliminary Estimated Construction Cost (10,000,000 IRR)")

# Histogram of a numerical variable
hist(train$V.2,
     main = "Histogram of V.2 (Total Floor Area)",
     xlab = "Total Floor Area (m^2)", ylab = "Frequency",
     col = "skyblue", border = "black")
abline(v = mean(train$V.2), col = "red", lwd = 2)  # vertical line for the mean
legend("topright", legend = paste("Mean:", round(mean(train$V.2), 2)),
       col = "red", lwd = 2)

# Boxplots to identify outliers
par(mfrow = c(1, 2))
boxplot(train$V.2, main = "V.2 Total Floor Area (m^2)", col = "skyblue", pch = 19)
boxplot(train$V.4, main = "V.4 Total Prelim. Cost (10,000,000 IRR)", col = "skyblue", pch = 19)
par(mfrow = c(1, 1))

# Scatter plot
plot(train$V.2, train$V.4,
     main = "Total Floor Area vs Preliminary Estimated Construction Cost",
     xlab = "Total Floor Area (m^2)",
     ylab = "Preliminary Estimated Construction Cost (10,000,000 IRR)",
     col = "blue", pch = 19)

# Bar plot of the locality category
barplot(table(train$V.1),
        main = "Bar Plot of Project Locality",
        xlab = "Project Locality", ylab = "Frequency",
        col = "skyblue")

# Actual cost against the main predictor (V.5 has the strongest relationship with Y)
plot(train$V.5, train$Y,
     main = "Preliminary Cost per Unit (V.5) vs Actual Construction Cost (Y)",
     xlab = "V.5 (10,000 IRR)", ylab = "Y Actual Construction Cost (10,000 IRR)",
     col = "blue", pch = 19)

# Actual cost against construction duration (points, not a line: rows are
# unordered projects, not a time series)
plot(train$V.7, train$Y,
     main = "Duration of Construction vs Actual Construction Cost",
     xlab = "Duration of Construction (Months)",
     ylab = "Actual Construction Cost (10,000 IRR)",
     col = "red", pch = 19)

# Density plot
plot(density(train$V.8),
     main = "Density Plot of Price per Unit Area (V.8)",
     xlab = "Price per Unit Area (10,000 IRR)", col = "blue")

pairs(train[, setdiff(num_vars, "V.1")])

#################### Data Preparation Code ######################

# Issues / improvements identified in the data understanding phase:
#  1. One completely blank row  -> removed before partitioning (above).
#  2. V.1 is a category coded as a number -> excluded from the numeric
#     predictors (it has 20 levels for 368 rows, which would add 19 sparse
#     dummy variables; a sensitivity check is shown below).
#  3. Predictors are on very different scales (V.2 in m^2, V.8 in 10,000 IRR)
#     -> standardised so the ridge/lasso penalty treats them equally.
#  4. Skewed variables with large values (V.2, V.4) -> checked as outliers.

### IQR Method - to identify outliers (limits taken from the TRAINING set) ###
iqr_limits <- function(x) {
  q <- quantile(x, c(0.25, 0.75))
  c(lower = q[[1]] - 1.5 * (q[[2]] - q[[1]]), upper = q[[2]] + 1.5 * (q[[2]] - q[[1]]))
}

outlier_summary <- sapply(c("V.2", "V.4", "Y"), function(v) {
  lim <- iqr_limits(train[[v]])
  c(lower = lim[["lower"]], upper = lim[["upper"]],
    n_outliers = sum(train[[v]] < lim[["lower"]] | train[[v]] > lim[["upper"]]))
})
print(round(outlier_summary, 1))

# Outliers here are real, large projects (not data errors), and the model must
# still be able to cost them, so the rows are KEPT. Deleting them would also
# make the training data unrepresentative of the test set. Standardisation plus
# regularisation limits the influence of extreme values.

### Predictor matrices ###
# Predictors: V.2 - V.8 (V.1 excluded). The target Y is NOT a predictor.
predictors <- c("V.2", "V.3", "V.4", "V.5", "V.6", "V.7", "V.8")

x_train_raw <- as.matrix(train[, predictors])
y_train     <- train$Y
x_test_raw  <- as.matrix(test[, predictors])
y_test      <- test$Y

## Standardizing the predictor variables ##
# Centre and scale using the TRAINING mean and SD only, then apply the SAME
# values to the test set. Scaling the test set by its own mean/SD would leak
# test information and shift the predictions.
x_train <- scale(x_train_raw)
train_center <- attr(x_train, "scaled:center")
train_scale  <- attr(x_train, "scaled:scale")
x_test <- scale(x_test_raw, center = train_center, scale = train_scale)

anyNA(x_train); anyNA(x_test)   # both FALSE
round(colMeans(x_train), 3)     # ~0 by construction
round(colMeans(x_test), 3)      # not exactly 0: correct, uses training statistics

# Variant without V.3 (Lot area) for one of the comparisons
x_train_excl_V3 <- x_train[, colnames(x_train) != "V.3"]
x_test_excl_V3  <- x_test[, colnames(x_test) != "V.3"]

##################  Modelling Code ###################

# glmnet: alpha = 0 is ridge, alpha = 1 is lasso, in between is elastic net.
# Standardisation was done above, so standardize = FALSE.
# cv.glmnet performs k-fold cross-validation on the TRAINING set to pick lambda.
# The same fold assignment (foldid) is used for every model so that their
# cross-validated errors are directly comparable.

nfolds <- 10
set.seed(2)
foldid <- sample(rep(1:nfolds, length.out = nrow(x_train)))

fit_cv <- function(x, alpha) {
  cv.glmnet(x, y_train, alpha = alpha, foldid = foldid, standardize = FALSE)
}

# Model 1 - Ridge, all predictors
cv_ridge <- fit_cv(x_train, alpha = 0)
# Model 2 - Lasso, all predictors
cv_lasso <- fit_cv(x_train, alpha = 1)
# Model 3 - Ridge, V.3 excluded
cv_ridge_excl_V3 <- fit_cv(x_train_excl_V3, alpha = 0)
# Model 4 - Lasso, V.3 excluded
cv_lasso_excl_V3 <- fit_cv(x_train_excl_V3, alpha = 1)

plot(cv_ridge, main = "Model 1 Ridge");            plot(cv_lasso, main = "Model 2 Lasso")
plot(cv_ridge_excl_V3, main = "Model 3 Ridge excl. V.3")
plot(cv_lasso_excl_V3, main = "Model 4 Lasso excl. V.3")

# Coefficient paths: how coefficients shrink as lambda increases
plot(cv_ridge$glmnet.fit, xvar = "lambda", label = TRUE)
plot(cv_lasso$glmnet.fit, xvar = "lambda", label = TRUE)

# The vertical dotted lines in the cv plots mark lambda.min (lowest
# cross-validated error) and lambda.1se (simplest model within one standard
# error of the minimum).

# Baseline: ordinary least squares on the same predictors, evaluated by the
# same cross-validation folds
ols_cv_rmse <- sqrt(mean(sapply(1:nfolds, function(k) {
  tr <- foldid != k
  fit <- lm(y_train[tr] ~ ., data = as.data.frame(x_train[tr, ]))
  mean((y_train[!tr] - predict(fit, as.data.frame(x_train[!tr, ])))^2)
})))

### Cross-validated results (training data only) ###
cv_rmse <- function(cv) sqrt(cv$cvm[cv$lambda == cv$lambda.min])

model_comparison <- data.frame(
  Model = c("Ridge", "Lasso", "Ridge excl. V.3", "Lasso excl. V.3", "OLS (baseline)"),
  Lambda_min = c(cv_ridge$lambda.min, cv_lasso$lambda.min,
                 cv_ridge_excl_V3$lambda.min, cv_lasso_excl_V3$lambda.min, NA),
  CV_RMSE = c(cv_rmse(cv_ridge), cv_rmse(cv_lasso),
              cv_rmse(cv_ridge_excl_V3), cv_rmse(cv_lasso_excl_V3), ols_cv_rmse)
)
model_comparison$CV_MSE <- model_comparison$CV_RMSE^2
print(model_comparison, digits = 4)

# The final model is chosen on cross-validated RMSE alone. The test set is not
# used to choose a model.
best_index <- which.min(model_comparison$CV_RMSE[1:4])
best_name  <- model_comparison$Model[best_index]
best_cv    <- list(cv_ridge, cv_lasso, cv_ridge_excl_V3, cv_lasso_excl_V3)[[best_index]]
best_x_test <- if (grepl("V.3", best_name)) x_test_excl_V3 else x_test
cat("Selected model:", best_name, "\n")
coef(best_cv, s = "lambda.min")

####################  Final Evaluation ####################
# The test set is used ONCE, for the selected model.

predictions <- as.numeric(predict(best_cv, newx = best_x_test, s = "lambda.min"))

# Compute metrics in the original units of Y (10,000 IRR)
errors <- y_test - predictions
test_MSE  <- mean(errors^2)
test_RMSE <- sqrt(test_MSE)
test_MAE  <- mean(abs(errors))
test_R2   <- 1 - sum(errors^2) / sum((y_test - mean(y_test))^2)

# Baseline for context: always predict the training mean
baseline_RMSE <- sqrt(mean((y_test - mean(y_train))^2))

# +/- 500,000 IRR check (= +/- 50 in units of Y)
within_threshold <- abs(errors) <= TOLERANCE
num_within_threshold <- sum(within_threshold)
percentage_within_threshold <- 100 * num_within_threshold / length(y_test)

results <- data.frame(
  Method = c(paste(best_name, "(CV, training)"), paste(best_name, "(test)"), "Mean baseline (test)"),
  RMSE = c(min(model_comparison$CV_RMSE[1:4]), test_RMSE, baseline_RMSE),
  MAE = c(NA, test_MAE, mean(abs(y_test - mean(y_train)))),
  R2 = c(NA, test_R2, NA),
  Within_500k_IRR = c(NA, percentage_within_threshold,
                      100 * mean(abs(y_test - mean(y_train)) <= TOLERANCE))
)
print(results, digits = 4)

cat("Test RMSE (10,000 IRR):", round(test_RMSE, 2),
    " = ", round(test_RMSE * 10000), "IRR\n")
cat("Test MAE  (10,000 IRR):", round(test_MAE, 2), "\n")
cat("Test R-squared:", round(test_R2, 3), "\n")
cat("Predictions within +/- 500,000 IRR (+/- 50):", num_within_threshold, "of",
    length(y_test), "=", round(percentage_within_threshold, 1), "%\n")

############## Plots of test set predictions vs actual values ######################

# Predicted vs actual with the +/- tolerance band around the perfect-fit line
plot(predictions, y_test, pch = 19,
     col = ifelse(within_threshold, "blue", "red"),
     xlab = "Predicted Y (10,000 IRR)", ylab = "Actual Y (10,000 IRR)",
     main = "Test Set: Predicted vs Actual")
abline(0, 1, col = "black")
abline(TOLERANCE, 1, col = "green", lty = 2)
abline(-TOLERANCE, 1, col = "green", lty = 2)
legend("topleft", legend = c("Within +/- 500,000 IRR", "Outside", "Perfect fit", "+/- tolerance"),
       col = c("blue", "red", "black", "green"), pch = c(19, 19, NA, NA),
       lty = c(NA, NA, 1, 2), cex = 0.8)

# Actual vs predicted for each test project, ordered by actual cost
ord <- order(y_test)
plot(seq_along(ord), y_test[ord], type = "l", col = "red", lwd = 2,
     xlab = "Test projects (ordered by actual cost)", ylab = "Y (10,000 IRR)",
     main = "Actual vs Predicted Construction Cost")
lines(seq_along(ord), predictions[ord], col = "blue", lty = 2, lwd = 2)
legend("topleft", legend = c("Actual", "Predicted"), col = c("red", "blue"), lty = 1:2, cex = 0.8)

# Bar plot of the proportion of predictions within the threshold
barplot(table(factor(within_threshold, levels = c(FALSE, TRUE),
                     labels = c("Outside Threshold", "Within Threshold"))),
        col = c("red", "green"),
        main = "Predictions Within +/- 500,000 Iranian Rial",
        ylab = "Frequency")

# Residuals: are the errors unbiased and evenly spread?
plot(predictions, errors, pch = 19, col = "blue",
     xlab = "Predicted Y (10,000 IRR)", ylab = "Residual (Actual - Predicted)",
     main = "Test Set Residuals")
abline(h = c(-TOLERANCE, 0, TOLERANCE), col = c("green", "black", "green"), lty = c(2, 1, 2))

############## Sensitivity checks (informational, do not select the model) ##########
# Does adding V.1 locality as dummy variables help? Compared on the same
# cross-validation folds used for the training data.
x_train_loc <- model.matrix(~ V.1, train)[, -1]
x_train_loc <- cbind(x_train, scale(x_train_loc))
cv_lasso_loc <- fit_cv(x_train_loc, alpha = 1)
cat("CV RMSE lasso without V.1:", round(cv_rmse(cv_lasso), 2),
    "| with V.1 dummies:", round(cv_rmse(cv_lasso_loc), 2), "\n")
