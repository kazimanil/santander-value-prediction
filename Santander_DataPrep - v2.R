## Worse RMSE Scores.

# Data Input ----
train = fread("train.csv")
test  = fread("test.csv")
sample = fread("sample_submission.csv")

# Libraries ----
library("stats")      # Factor Analysis
library("psych")      # Factor Analysis -- stats::princomp has a better interface.
library("xgboost")    # Gradient Boosting
library("MlBayesOpt") # Optimisation for Gradient Boosting.
library("Matrix")     # To Prepare the data for XGBoost model
library("MASS")       # For box-cox transformation
source("C:/Users/kazimanil/Documents/Digitallency_GGPlot_Theme.R")

# Data Manipulation ----
# In this dataset, I will try a different approach, I will use pca on train set only. Then apply the results on test set as well. 
sanity_check =
  data.table(colname  = colnames(train[, 3:ncol(train)]),
             train_sd = as.data.table(sapply(X = train[, 3:ncol(train)], sd, na.rm = TRUE))$V1)

colstokeep = append("ID", 
                      append("target", sanity_check[train_sd != 0]$colname))
train  = train[, colstokeep, with = FALSE]

colstokeep = append("ID", sanity_check[train_sd != 0]$colname)
test   = test[, colstokeep, with = FALSE]

train = cbind(train[, 1:2],
              sapply(train[, 3:ncol(train)], as.numeric))
rm(sanity_check, colstokeep)

# Next step should be to use PCA to combine variables into new variables.
dt_pca = train[, 3:ncol(train)] # Numeric Columns
dt_test = test[, 2:ncol(test)]  # Numeric Columns
n_cols = ncol(dt_pca)  # Number of columns
dt_cor = cor(dt_pca)   # Correlation Matrix
dt_eig = eigen(dt_cor) # EigenValues & EigenVectors
dt_eiv = dt_eig$values # EigenValues

plot(dt_eiv[1:50],
     main="Scree Plot",
     ylab="Eigenvalues",
     xlab="Component number",
     type='b')
abline(h=1, lty=2)

n_facs = 20 # Based on scree-plot I chose n_facs to be 20.
pca_results = psych::principal(r = dt_cor, nfactors = n_facs, rotate = "varimax", scores = TRUE, n.obs = nrow(dt_pca))
pca_weights = as.matrix(pca_results$loadings)
class(pca_weights) = "matrix"
pca_weights = as.data.table(pca_weights)
pca_weights = sapply(pca_weights, FUN = function(x) ifelse(x < 0.3 & x > -0.3, 0, x), simplify = TRUE)

# Now let's construct new data.tables:
new_train = as.matrix(dt_pca) %*% pca_weights
new_test  = as.matrix(dt_test[1:342]) %*% pca_weights
for(i in 1:49){ ## cannot allocate vector size, otherwise.
  startrow  = 343 + ((i-1)* 1000)
  endrow    = 342 + (i * 1000)
  new_test  = rbind(new_test, 
                    as.matrix(dt_test[startrow:endrow]) %*% pca_weights)
}

# Lets Name new factors 
colnames   = as.character(seq(1, n_facs, 1))
colnames   = paste0("factor", colnames)
new_train  = as.data.table(new_train)
new_train  = `colnames<-`(new_train, colnames)
new_test  = as.data.table(new_test)
new_test  = `colnames<-`(new_test, colnames)

# Clearence
save(pca_weights, pca_results, dt_eiv, dt_cor, dt_eig, file = "model2_weights.RData")
rm(endrow, startrow, n_facs, n_cols, i, dt_cor, dt_eig, dt_eiv, pca_results, pca_weights, dt_pca, dt_test); gc()

# It's sanity check time.
sanity_check = data.table(factor = colnames,
                          sd     = sapply(new_train, sd, simplify = TRUE, USE.NAMES = TRUE),
                          mean   = sapply(new_train, mean, simplify = TRUE, USE.NAMES = TRUE))

sanity_check = sanity_check[sd == 0 & mean == 0]
cols_to_keep = colnames[!colnames %in% sanity_check$factor]

# length(cols_to_keep) == length(colnames) is TRUE so keep ALL.
new_train = cbind(train[, 1:2], new_train)
new_test  = cbind(test[, 1], new_test)
save(new_test, new_train, file = "PCA-v2.RData")

# Analysis ----
# Let's check the distribution of target first
ggplot(data = new_train, aes(x = target)) + 
  geom_histogram(col = "Darkorange", fill = "Turquoise")  +
  theme_dt() + labs(x = "Target", y = "Frequency")

# We have a very skewed target. We shall normalise it before putting it into xGBoost. (ref: http://rcompanion.org/handbook/I_12.html)
box = boxcox(new_train$target ~ 1, 
             lambda = seq(-1, 1, 0.01))
cox = data.table(lambda = box$x, 
                 logLL  = box$y)[order(-logLL)]
lambda = cox[1]$lambda
new_train[, target_boxcox := (target ^ lambda - 1) / lambda]; rm(box, cox);
shapiro.test(new_train$target_boxcox) # ref: http://www.sthda.com/english/wiki/normality-test-in-r

ggplot(data = new_train, aes(x = target_boxcox)) + 
  geom_histogram(aes(y = ..density..), col = "Darkorange", fill = "Turquoise", bins = 30) + 
  geom_density(col = "red", size = 1) +
  theme_dt() +
  labs(x = "Target (Box-Cox Transformation)", y = "Frequency")

# xGBoost ----

# Data Preparation
my_nfolds  = as.integer(13);
xgb_matrix = as.matrix(new_train[, 3:ncol(new_train)]);
xgb_matrix = xgb.DMatrix(xgb_matrix, label = xgb_matrix[, 21]);
new_test$target_boxcox = 0
xgb_tester = as.matrix(new_test[, 2:ncol(new_test)])
xgb_tester = xgb.DMatrix(data = xgb_tester, label = xgb_tester[, 21]);

p <- list(objective = "reg:linear",
          booster = "gbtree",
          eval_metric = "rmse",
          eta = 0.01,
          max_depth = 6,
          min_child_weight = 20,
          gamma = 0.9,
          subsample = 0.9)

m_xgb2 <- xgboost(params = p, 
                  data = xgb_matrix, 
                  nrounds = 333, 
                  print_every_n = 100, 
                  early_stopping_rounds = 100)

new_test[, target := (target_boxcox * lambda + 1) ^ ( 1 / lambda)]
submission_xgb1 = new_test[, c(1,23)]
fwrite(submission_xgb1, file = "submission_xgb2.csv")
