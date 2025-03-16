# Load necessary libraries
library(tidyverse)
library(caret)
library(car)

# Load data
file_path <- "C:/Users/Sensorbtf/Nauka-O-Danych/Data.CSV"
data <- read.csv(file_path)

# Identify and remove single-level factors
single_level_cols <- sapply(data, function(col) length(unique(col)) == 1)
data <- data[, !single_level_cols]

# Convert categorical variables to factors
data <- data %>%
  mutate(across(where(is.character), as.factor))

# Ensure the target variable `val` is not part of predictors
target_col <- "val"
X <- data %>%
  select(-one_of(target_col, "upper", "lower")) # Exclude target and other irrelevant columns
y <- data[[target_col]]

# Split data into training and test sets
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Linear Regression Model
model_lm <- lm(y_train ~ ., data = as.data.frame(X_train))
print(summary(model_lm))

# Predict and evaluate Linear Regression
y_pred_lm <- predict(model_lm, newdata = as.data.frame(X_test))
mse_lm <- mean((y_test - y_pred_lm)^2)
cat("Linear Regression - MSE:", mse_lm, "\n")

# Ridge Regression Model
model_ridge <- train(
  y_train ~ ., data = as.data.frame(X_train),
  method = "ridge",
  trControl = trainControl(method = "cv", number = 5),
  preProcess = c("center", "scale") # Ensure data scaling
)

# Predict and evaluate Ridge Regression
y_pred_ridge <- predict(model_ridge, newdata = as.data.frame(X_test))
mse_ridge <- mean((y_test - y_pred_ridge)^2)
cat("Ridge Regression - MSE:", mse_ridge, "\n")
