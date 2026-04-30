# ========================
# 1. Load libraries
# ========================
library(caret)
library(glmnet)
library(rpart)
library(kknn)
library(lightgbm)
library(randomForest)
library(xgboost)
library(e1071)
library(nnet)
library(ggplot2)
library(Metrics)
library(dplyr)
library(ggpubr)
set.seed(123)

# ========================
# 2. Read data and select columns
# ========================
setwd("your workplace")

amino_acid_data <- read.csv("df.csv")


# ========================
# 3. Split into training (80%) and testing (20%) sets
# ========================
set.seed(1234)
trainIndex <- createDataPartition(amino_acid_data$age_year, p = 0.8, list = FALSE)
train_data <- amino_acid_data[trainIndex, ]
test_data  <- amino_acid_data[-trainIndex, ]

# Save training and testing sets
write.csv(train_data, "train_data.csv", row.names = FALSE)
write.csv(test_data, "test_data.csv", row.names = FALSE)

# ========================
# 4. Standardize features
# ========================
preProc <- preProcess(train_data[, -1], method = c("center", "scale"))
train_scaled <- train_data
train_scaled[, -1] <- predict(preProc, train_data[, -1])

test_scaled <- test_data
test_scaled[, -1] <- predict(preProc, test_data[, -1])

# Save preprocessing object
saveRDS(preProc, "preProc_model.rds")

# ========================
# 5. Set up 5-fold cross-validation
# ========================
train_control <- trainControl(
  method = "cv",
  number = 5,
  returnResamp = "all",
  savePredictions = "final"
)

# Function to calculate R²
rsq <- function(true_values, predictions) {
  ss_res <- sum((true_values - predictions)^2)
  ss_tot <- sum((true_values - mean(true_values))^2)
  return(1 - ss_res/ss_tot)
}

# Function to calculate performance metrics
calculate_metrics <- function(true_values, predictions) {
  rmse_val <- rmse(true_values, predictions)
  mae_val  <- mae(true_values, predictions)
  r2_val   <- rsq(true_values, predictions)
  return(list(RMSE = rmse_val, MAE = mae_val, R2 = r2_val))
}

# ========================
# 6. Train models
# ========================
model_results <- list()
predictions_all <- list()

# ------------------------
# 6.1 Lasso Regression
# ------------------------
lasso_model <- train(
  age_year ~ ., data = train_scaled,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 0.2, length = 100))
)
lasso_pred <- predict(lasso_model, test_scaled)
model_results[["Lasso"]] <- calculate_metrics(test_scaled$age_year, lasso_pred)
predictions_all[["Lasso"]] <- data.frame(age_year = test_scaled$age_year, predicted = lasso_pred)

# ------------------------
# 6.2 Elastic Net
# ------------------------
enet_model <- train(
  age_year ~ ., data = train_scaled,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(alpha = 0.5, lambda = seq(0.0001, 0.2, length = 100))
)
enet_pred <- predict(enet_model, test_scaled)
model_results[["ElasticNet"]] <- calculate_metrics(test_scaled$age_year, enet_pred)
predictions_all[["ElasticNet"]] <- data.frame(age_year = test_scaled$age_year, predicted = enet_pred)

# ------------------------
# 6.3 Linear Regression
# ------------------------
lm_model <- train(age_year ~ ., data = train_scaled, method = "lm", trControl = train_control)
lm_pred <- predict(lm_model, test_scaled)
model_results[["LinearRegression"]] <- calculate_metrics(test_scaled$age_year, lm_pred)
predictions_all[["LinearRegression"]] <- data.frame(age_year = test_scaled$age_year, predicted = lm_pred)

# ------------------------
# 6.4 Decision Tree
# ------------------------
dt_model <- train(
  age_year ~ ., data = train_scaled,
  method = "rpart",
  trControl = train_control,
  tuneGrid = expand.grid(cp = seq(0.0001, 0.05, length = 50))
)
dt_pred <- predict(dt_model, test_scaled)
model_results[["DecisionTree"]] <- calculate_metrics(test_scaled$age_year, dt_pred)
predictions_all[["DecisionTree"]] <- data.frame(age_year = test_scaled$age_year, predicted = dt_pred)

# ------------------------
# 6.5 K-Nearest Neighbors
# ------------------------
knn_model <- train(
  age_year ~ ., data = train_scaled,
  method = "kknn",
  trControl = train_control,
  tuneLength = 20
)
knn_pred <- predict(knn_model, test_scaled)
model_results[["KNN"]] <- calculate_metrics(test_scaled$age_year, knn_pred)
predictions_all[["KNN"]] <- data.frame(age_year = test_scaled$age_year, predicted = knn_pred)

# ------------------------
# 6.6 Random Forest
# ------------------------
rf_model <- train(
  age_year ~ ., data = train_scaled,
  method = "ranger",
  trControl = train_control,
  tuneGrid = expand.grid(
    mtry = seq(2, ncol(train_scaled)-1, by = 2),
    splitrule = "variance",
    min.node.size = seq(3, 30, by = 3)
  ),
  num.trees = 500
)
rf_pred <- predict(rf_model, test_scaled)
model_results[["RandomForest"]] <- calculate_metrics(test_scaled$age_year, rf_pred)
predictions_all[["RandomForest"]] <- data.frame(age_year = test_scaled$age_year, predicted = rf_pred)

# ------------------------
# 6.7 XGBoost
# ------------------------
xgb_model <- train(
  age_year ~ ., data = train_scaled,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = expand.grid(
    nrounds = c(100, 150),
    max_depth = c(3, 6),
    eta = c(0.05, 0.1),
    gamma = c(0, 0.1),
    colsample_bytree = c(0.8, 1),
    min_child_weight = c(1, 3),
    subsample = c(0.8, 1)
  )
)
xgb_pred <- predict(xgb_model, test_scaled)
model_results[["XGBoost"]] <- calculate_metrics(test_scaled$age_year, xgb_pred)
predictions_all[["XGBoost"]] <- data.frame(age_year = test_scaled$age_year, predicted = xgb_pred)

# ------------------------
# 6.8 Support Vector Machine
# ------------------------
svm_model <- train(
  age_year ~ ., data = train_scaled,
  method = "svmRadial",
  trControl = train_control,
  tuneLength = 20
)
svm_pred <- predict(svm_model, test_scaled)
model_results[["SVM"]] <- calculate_metrics(test_scaled$age_year, svm_pred)
predictions_all[["SVM"]] <- data.frame(age_year = test_scaled$age_year, predicted = svm_pred)

# ------------------------
# 6.9 Neural Network
# ------------------------
nn_model <- train(
  age_year ~ ., data = train_scaled,
  method = "nnet",
  trControl = train_control,
  tuneLength = 20,
  linout = TRUE,
  maxit = 500,
  trace = FALSE
)
nn_pred <- predict(nn_model, test_scaled)
model_results[["NeuralNetwork"]] <- calculate_metrics(test_scaled$age_year, nn_pred)
predictions_all[["NeuralNetwork"]] <- data.frame(age_year = test_scaled$age_year, predicted = nn_pred)

# ------------------------
# 6.10 Lightgbm
# ------------------------
# 1. Define 5-fold CV splits
folds <- createFolds(train_scaled$age_year, k = 5)

# 2. Define hyperparameter grid
param_grid <- expand.grid(
  num_leaves = c(15, 31, 63),
  learning_rate = c(0.01, 0.03, 0.05),
  feature_fraction = c(0.7, 0.8, 0.9),
  bagging_fraction = c(0.7, 0.8, 0.9)
)

best_params <- NULL
best_rmse <- Inf

# 3. Loop over all parameter combinations for grid search
for(i in 1:nrow(param_grid)){
  
  params <- list(
    objective = "regression",
    metric = "rmse",
    num_leaves = param_grid$num_leaves[i],
    learning_rate = param_grid$learning_rate[i],
    feature_fraction = param_grid$feature_fraction[i],
    bagging_fraction = param_grid$bagging_fraction[i],
    bagging_freq = 5
  )
  
  fold_rmse <- c()
  
  # 4. Perform 5-fold CV
  for(f in seq_along(folds)){
    val_idx <- folds[[f]]
    train_idx <- setdiff(seq_len(nrow(train_scaled)), val_idx)
    
    dtrain <- lgb.Dataset(data = as.matrix(train_scaled[train_idx, -1]),
                          label = train_scaled$age_year[train_idx])
    dval <- lgb.Dataset(data = as.matrix(train_scaled[val_idx, -1]),
                        label = train_scaled$age_year[val_idx])
    
    model <- lgb.train(
      params = params,
      data = dtrain,
      nrounds = 1000,
      valids = list(val = dval),
      early_stopping_rounds = 50,
      verbose = -1
    )
    
    preds <- predict(model, as.matrix(train_scaled[val_idx, -1]), num_iteration = model$best_iter)
    rmse_val <- sqrt(mean((preds - train_scaled$age_year[val_idx])^2))
    fold_rmse <- c(fold_rmse, rmse_val)
  }
  
  mean_rmse <- mean(fold_rmse)
  
  if(mean_rmse < best_rmse){
    best_rmse <- mean_rmse
    best_params <- params
  }
}

# 5. Train final LightGBM model on full training set
dtrain_all <- lgb.Dataset(data = as.matrix(train_scaled[, -1]), label = train_scaled$age_year)
final_lgb_model <- lgb.train(
  params = best_params,
  data = dtrain_all,
  nrounds = 1000,
  verbose = 0,
  valids = list(train = dtrain_all),
  early_stopping_rounds = 50
)

# 6. Predict on test set
lgb_pred <- predict(final_lgb_model, as.matrix(test_scaled[, -1]), num_iteration = final_lgb_model$best_iter)

# 7. Calculate performance metrics
calculate_metrics <- function(true_values, predictions){
  rmse_val <- sqrt(mean((true_values - predictions)^2))
  mae_val <- mean(abs(true_values - predictions))
  r2_val <- cor(true_values, predictions)^2
  return(list(RMSE = rmse_val, MAE = mae_val, R2 = r2_val))
}

lgb_metrics <- calculate_metrics(test_scaled$age_year, lgb_pred)
model_results[["LightGBM"]] <- lgb_metrics

# 8. Save predictions for later use
predictions_all[["LightGBM"]] <- data.frame(
  age_year = test_scaled$age_year,
  predicted = lgb_pred
)

# 9. Print summary
print(best_params)
cat("Best RMSE from CV:", best_rmse, "\n")
print(lgb_metrics)

# ========================
# 7. Save model objects
# ========================
saveRDS(lasso_model, "lasso_model.rds")
saveRDS(enet_model, "elastic_net_model.rds")
saveRDS(lm_model, "lm_model.rds")
saveRDS(dt_model, "dt_model.rds")
saveRDS(knn_model, "knn_model.rds")
saveRDS(rf_model, "rf_model.rds")
saveRDS(xgb_model, "xgb_model.rds")
saveRDS(svm_model, "svm_model.rds")
saveRDS(nn_model, "nn_model.rds")
saveRDS(final_lgb_model, "lgb_model.rds")
# ========================
# 8. Save predictions
# ========================
for(name in names(predictions_all)){
  write.csv(predictions_all[[name]], paste0(name, "_test_predictions.csv"), row.names = FALSE)
}

# ========================
# 9. Summarize performance
# ========================
final_results <- do.call(rbind, lapply(model_results, function(x) unlist(x)))
final_results <- data.frame(Model = rownames(final_results), final_results, row.names = NULL)
write.csv(final_results, "model_test_performance.csv", row.names = FALSE)

