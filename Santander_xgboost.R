# Santander xGBoost Estimation ----
lambda = 0.12
test  = readRDS(file = "test_PCAv3_xgb.RDS")
train = readRDS(file = "train_PCAv3_xgb.RDS")

# Libraries ----
library("randomForest")
library("xgboost")

# Variable Selection ----
# There are still 456 variables in the dataset and I will utilise RF to find out the best performing among them.
# Since Random Forest does not assume normality, I will go with original dataset and not use Box-Cox transformation here.
rF_obj = randomForest(x = train[, 4:ncol(train)],
                      y = train$target,
                      xtest = test[, 2:ncol(test)],
                      ntree = 1000,
                      nodesize = 45, 
                      importance = TRUE)
col_select = data.table(columns = rownames(rF_obj$importance),
                        incmse  = rF_obj$importance[,1])[order(-incmse)]
plot(col_select[1:100]$incmse)
colstokeep = col_select[1:18]$columns
xgb_test   = xgb.DMatrix(as.matrix(test[, colstokeep, with = FALSE]))
# colstokeep = append("target", colstokeep)
xgb_train  = xgb.DMatrix(as.matrix(train[, colstokeep, with = FALSE]),
                         label = train$target_boxcox)

p <- list(objective = "reg:linear",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 8,
          eta = 0.0075,
          max_depth = 30,
          min_child_weight = 52,
          gamma = 0.09690536,
          subsample = 0.95,
          colsample_bytree = 0.1,
          colsample_bylevel = 0.1,
          alpha = 0,
          lambda = 100)

m_xgb <- xgboost(params = p, 
                 data = xgb_train, 
                 nrounds = 10000, 
                 print_every_n = 100, 
                 early_stopping_rounds = 300)

test$target_boxcox = predict(object = m_xgb, newdata = xgb_test)
test[, target := (target_boxcox * lambda + 1) ^ ( 1 / lambda)]
submission_xgb3 = test[, c(1,459)]
fwrite(submission_xgb3, file = "submission_xgb3.csv")
