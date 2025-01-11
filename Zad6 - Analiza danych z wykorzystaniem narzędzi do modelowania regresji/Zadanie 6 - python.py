# Load libraries
library(readr)
library(dplyr)
library(caret)
library(glmnet)
library(keras)
library(ggplot2)
library(car)
library(stats)

# Load data
file_path <- "C:/Users/Sensorbtf/OneDrive/Studia/Magisterka/Dane/IHME_GBD_2019_SMOKING_AGE_1990_2019_INIT_AGE_Y2021M05D27.CSV"
data <- read_csv(file_path)

# Encode categorical variables
data_encoded <- data %>%
  mutate(across(c(measure_name, location_name, sex_name, `age_group_name.1`), as.factor)) %>%
  model.matrix(~ . - 1, data = .) %>% as.data.frame()

# Prepare variables
X <- data_encoded %>% select(-val, -upper, -lower)
y <- data_encoded$val

# Standardize data
scaler <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(scaler, X)

# Split data into training and test sets
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_scaled[train_index, ]
X_test <- X_scaled[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Linear Regression (Normalized Data)
lr <- lm(y_train ~ ., data = data.frame(X_train, y_train))
y_pred_lr <- predict(lr, newdata = data.frame(X_test))

# Ridge Regression (Normalized Data)
ridge <- cv.glmnet(as.matrix(X_train), y_train, alpha = 0)
y_pred_ridge <- predict(ridge, as.matrix(X_test), s = "lambda.min")

# Neural Network (Normalized Data)
model_nn <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

model_nn %>% compile(
  optimizer = "adam",
  loss = "mean_squared_error",
  metrics = c("mse")
)

history <- model_nn %>% fit(
  as.matrix(X_train), y_train,
  epochs = 100,
  batch_size = 16,
  verbose = 1
)

y_pred_nn <- model_nn %>% predict(as.matrix(X_test))

# Evaluate models
evaluate_model <- function(y_true, y_pred) {
  r2 <- cor(y_true, y_pred)^2
  mse <- mean((y_true - y_pred)^2)
  mae <- mean(abs(y_true - y_pred))
  list(R2 = r2, MSE = mse, MAE = mae)
}

cat("Linear Regression (Normalized Data):\n")
print(evaluate_model(y_test, y_pred_lr))

cat("\nRidge Regression (Normalized Data):\n")
print(evaluate_model(y_test, y_pred_ridge))

cat("\nNeural Network (Normalized Data):\n")
print(evaluate_model(y_test, y_pred_nn))

# Residual analysis for Linear Regression (Normalized Data)
residuals <- y_test - y_pred_lr

# Histogram of residuals
ggplot(data.frame(residuals), aes(residuals)) +
  geom_histogram(bins = 20, fill = "blue", alpha = 0.7) +
  ggtitle("Histogram of Residuals") +
  xlab("Residuals") +
  ylab("Frequency")

# Shapiro-Wilk Test for residuals
shapiro_test <- shapiro.test(residuals)
cat("\nShapiro-Wilk Test for Residuals:\n")
print(shapiro_test)

# Q-Q Plot for residuals
qqPlot(residuals, main = "Q-Q Plot of Residuals")

# Durbin-Watson Test for residuals
dw_test <- durbinWatsonTest(lm(y_test ~ y_pred_lr))
cat("\nDurbin-Watson Statistic for Residuals:\n")
print(dw_test)
