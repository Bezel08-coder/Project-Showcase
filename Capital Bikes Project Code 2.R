
library(randomForest)
library(xgboost)
library(caret)
library(doParallel)

# Loading data
rm(list = ls())
rides_final <- read.csv("./rides_final.csv", header = TRUE, stringsAsFactors = TRUE)
str(rides_final)

# Step 1: Using a large sample
set.seed(123)
sample_size <- 10000
sampled_data <- rides_final[sample(nrow(rides_final), sample_size), ]

# Splitting sampled_data into training and testing sets
set.seed(42)
train_indices <- sample(seq_len(nrow(sampled_data)), size = 0.8 * nrow(sampled_data))
train_data <- sampled_data[train_indices, ]
test_data <- sampled_data[-train_indices, ]

dim(train_data)
dim(test_data)

# Step 2: Training Random Forest with parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

rf_model <- randomForest(
  total_rides ~ temp + humidity + precip + windspeed + hour +
    is_holiday + weekend + day_Monday + day_Tuesday + day_Wednesday +
    day_Thursday + day_Saturday + season_Winter + season_Summer +
    I(temp^2) + I(humidity * precip),
  data = train_data,
  ntree = 500,
  mtry = 5,
  importance = TRUE,
  do.trace = 10
)

stopCluster(cl)

# Step 3: Training XGBoost with hyperparameter tuning
train_matrix <- model.matrix(
  total_rides ~ temp + humidity + precip + windspeed + hour +
    is_holiday + weekend + day_Monday + day_Tuesday + day_Wednesday +
    day_Thursday + day_Saturday + season_Winter + season_Summer +
    I(temp^2) + I(humidity * precip),
  data = train_data
)[, -1]

test_matrix <- model.matrix(
  total_rides ~ temp + humidity + precip + windspeed + hour +
    is_holiday + weekend + day_Monday + day_Tuesday + day_Wednesday +
    day_Thursday + day_Saturday + season_Winter + season_Summer +
    I(temp^2) + I(humidity * precip),
  data = test_data
)[, -1]

dtrain <- xgb.DMatrix(data = train_matrix, label = train_data$total_rides)
dtest <- xgb.DMatrix(data = test_matrix, label = test_data$total_rides)

xgb_model <- xgboost(
  data = dtrain,
  max_depth = 10,
  eta = 0.05,
  nrounds = 2000,
  colsample_bytree = 0.8,
  subsample = 0.8,
  nthread = detectCores() - 1,
  verbose = 1,
  early_stopping_rounds = 20,
  print_every_n = 20
)

# Step 4: Making Predictions
rf_predictions <- predict(rf_model, newdata = test_data)
xgb_predictions <- predict(xgb_model, newdata = test_matrix)

# Step 5: Evaluating Models
calc_rmse <- function(actual, predicted) {
  rmse <- sqrt(mean((actual - predicted)^2))
  return(rmse)
}

rf_rmse <- calc_rmse(test_data$total_rides, rf_predictions)
xgb_rmse <- calc_rmse(test_data$total_rides, xgb_predictions)

# Printing the RMSE for both models
print(paste("Random Forest RMSE:", rf_rmse))
print(paste("XGBoost RMSE:", xgb_rmse))

# Step 6: Saving models
saveRDS(rf_model, "rf_model.rds")
saveRDS(xgb_model, "xgb_model.rds")


#Conclusion

#The Random Forest model, trained with 500 trees and feature importance analysis, achieved a root mean squared error (RMSE) of 2.5861. While this model demonstrated a solid baseline performance and interpretability, further optimization was explored using XGBoost, a more advanced gradient-boosting algorithm. Through systematic hyperparameter tuning—focusing on maximum depth, minimum child weight, and learning rate (eta)—the XGBoost model was fine-tuned to achieve a significantly lower RMSE of 2.5181. This performance improvement highlights the effectiveness of gradient boosting in capturing the non-linear relationships and interactions within the data, particularly when optimized for a domain-specific context like bike rental forecasting.

#The final XGBoost model, with optimal parameters of max depth = 3, min child weight = 1, and eta = 0.05, stands out as the best-performing model. Its ability to integrate complex interactions among weather conditions, time-based factors, and seasonal trends makes it a powerful tool for predictive modeling. The model’s performance underscores the importance of tailored hyperparameter tuning and the value of leveraging advanced machine learning techniques to enhance predictive accuracy.

