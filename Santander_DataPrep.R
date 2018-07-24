# Data Input ----
train = fread("train.csv")
test  = fread("test.csv")
sample = fread("sample_submission.csv")

# Libraries ----
library("stats")      # Factor Analysis
#library("psych")     # Factor Analysis -- stats::princomp has a better interface.
library("xgboost")    # Gradient Boosting
library("MlBayesOpt") # Optimisation for Gradient Boosting.

# Data Manipulation ----
# Since there are lots of variables I will have to make a selection. Thereby, I will first decide if variables distribute differently in test and train sets assuming they distribute normally. I will use student's t-distribution and %99 confidence interval for this test. Then I will filter out variables which fail the test.

summary =
  data.table( colname    = colnames(train[, 3:ncol(train)]),
              train_sd   = as.data.table(sapply(train[, 3:ncol(train)], sd, na.rm = TRUE))$V1,
              train_mean = as.data.table(sapply(train[, 3:ncol(train)], mean, na.rm = TRUE))$V1,
              train_rows = nrow(train),
              test_sd    = as.data.table(sapply(test[, 2:ncol(test)], sd, na.rm = TRUE))$V1,
              test_mean  = as.data.table(sapply(test[, 2:ncol(test)], mean, na.rm = TRUE))$V1,
              test_rows  = nrow(test))
summary[, t_value := (train_mean - test_mean) / sqrt((train_sd^2/train_rows) + (test_sd^2 /test_rows))]
summary[, t_pos := qt(p = 0.99, df = nrow(test) + nrow(train) - 2)]
summary[, t_neg := -qt(p = 0.99, df = nrow(test) + nrow(train) - 2)]
selected_cols = summary[t_value >= t_neg & t_value <= t_pos]$colname
train = train[, append("ID", append("target", selected_cols)), with = FALSE]
test  = test[, append("ID",selected_cols), with = FALSE]

# More than half of the variables cannot pass. After deciding which columns to keep I will merge it with the original data. However, some columns have type mismatch. So I will turn all of them into numeric variables ( rather then integer in some cases).

train = cbind(train[, 1:2],
              sapply(train[, 3:ncol(train)], as.numeric))
test  = cbind(test[, 1],
              sapply(test[, 2:ncol(test)], as.numeric))
rm(summary, selected_cols)

# Next step should be to use PCA to combine variables into new variables.
dt_pca = rbind(train[, 3:ncol(train)], test[, 2:ncol(test)])
n_cols = ncol(dt_pca)  # Number of columns
dt_cor = cor(dt_pca)   # Correlation Matrix
dt_eig = eigen(dt_cor) # EigenValues & EigenVectors
dt_eiv = dt_eig$values # EigenValues
n_facs = length(dt_eiv[dt_eiv > 1]) # Amount of Factors to be derived based on "EigenValue > 1" criterion. Let's depict this with a graph as well.

plot(dt_eiv,
     main="Scree Plot",
     ylab="Eigenvalues",
     xlab="Component number",
     type='b')
abline(h=1, lty=2)

pca_results = psych::principal(r = dt_cor, nfactors = n_facs, rotate = "varimax", scores = TRUE, n.obs = nrow(dt_pca))
pca_weights = as.matrix(pca_results$loadings)
class(pca_weights) = "matrix"
pca_weights = as.data.table(pca_weights)
pca_weights = sapply(pca_weights, FUN = function(x) ifelse(x < 0.3 & x > -0.3, 0, x), simplify = TRUE)
# pca_results  = prcomp(x = dt_pca, retx = TRUE, scale = TRUE, tol = 0.5)
# pca_results2 = princomp(x = dt_pca)

# Now let's construct new data.tables:
new_dt     = as.matrix(dt_pca) %*% pca_weights
colnames   = as.character(seq(1, n_facs, 1))
colnames   = paste0("factor", colnames)
new_dt     = as.data.table(new_dt)
new_dt     = `colnames<-`(new_dt, colnames)

rm(dt_cor, dt_eig, dt_eiv, pca_results, pca_weights); gc()
new_train  = cbind(train[, 1:2], 
                   new_dt[1:nrow(train), 1:n_facs])
test_start = nrow(train)+1
test_end   = nrow(test)+nrow(train)
new_test   = cbind(test[, 1], 
                   new_dt[test_start:test_end, 1:n_facs])
rm(test_start, test_end, n_cols, n_facs, colnames, dt_pca, test, train, new_dt)
save(new_test, new_train, file = "PCA.RData")

#xGBoost -- The model parameters are obtained with bayesian optimization package (https://www.kaggle.com/kailex/santander-eda-features)
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
          lambda = 100,
          nrounds = 10000)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 100, early_stopping_rounds = 300)
