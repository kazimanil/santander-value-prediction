## In this version, I will have another approach for PCA.
# Data Input ----
rm(list = ls()); gc()
train = fread("train.csv")
test  = fread("test.csv")
sample = fread("sample_submission.csv")

# Libraries ----
library("Hmisc") # For Correlation Matrix
library("MASS")  # For Box-Cox Transformation

# Data Manipulation ----
# I will delete columns without variation.
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

# I will normalise the target variable
box = boxcox(train$target ~ 1, 
             lambda = seq(-1, 1, 0.01))
cox = data.table(lambda = box$x, 
                 logLL  = box$y)[order(-logLL)]
lambda = cox[1]$lambda
train[, target_boxcox := (target ^ lambda - 1) / lambda]; rm(box, cox);

# Correlation Matrix ----
# I will check the correlations with original target variable. Then choose the best performing variables as an input to PCA.
cor_dta = as.matrix(train[, 2:4737]) # Excluding ID and Target_BoxCox for Correlation Matrix.
cor_mtx = rcorr(x = cor_dta, type = "pearson") 
cor_chk = data.table(
  variable = rownames(cor_mtx$r),
  strength = as.data.table(cor_mtx$r)$target,
  p.values = as.data.table(cor_mtx$P)$target
  )
cor_chk[, significant := ifelse(p.values < 0.05, 1, 0)]
sig.vars = cor_chk[significant == 1]$variable
colstokeep = append("ID", append("target", append("target_boxcox", sig.vars)))
train = train[, colstokeep, with = FALSE]
colstokeep = append("ID", sig.vars)
test  = test[, colstokeep, with = FALSE]
rm(colstokeep, cor_mtx, cor_dta)

# Principal Components Analysis ----
# Now selected variables will be put into PCA.
dt_pca = rbind(train[, 4:ncol(train)], test[, 2:ncol(test)])
n_cols = ncol(dt_pca)  # Number of columns
dt_cor = cor(dt_pca)   # Correlation Matrix
dt_eig = eigen(dt_cor) # EigenValues & EigenVectors
dt_eiv = dt_eig$values # EigenValues
n_facs = length(dt_eiv[dt_eiv > 1]) # Amount of Factors to be derived based on "EigenValue > 1" criterion. Let's depict this with a graph as well.
sp_fac = 5

plot(dt_eiv[1:50],
     main="Scree Plot",
     ylab="Eigenvalues",
     xlab="Component number",
     type='b')
abline(h=1, lty=2)

# PCA for RF & xGBoost ----
pca_results = psych::principal(r = dt_cor, nfactors = n_facs, rotate = "varimax", scores = TRUE, n.obs = nrow(dt_pca))
pca_weights = as.matrix(pca_results$loadings)
class(pca_weights) = "matrix"
pca_weights = as.data.table(pca_weights)
pca_weights = sapply(pca_weights, FUN = function(x) ifelse(x < 0.3 & x > -0.3, 0, x), simplify = TRUE)

# Now let's construct new data.tables:
new_dt     = as.matrix(dt_pca) %*% pca_weights
colnames   = as.character(seq(1, n_facs, 1))
colnames   = paste0("factor", colnames)
new_dt     = as.data.table(new_dt)
new_dt     = `colnames<-`(new_dt, colnames)

# It's sanity check time.
sanity_check = data.table(factor = colnames,
                          sd     = sapply(new_dt, sd, simplify = TRUE, USE.NAMES = TRUE),
                          mean   = sapply(new_dt, mean, simplify = TRUE, USE.NAMES = TRUE))

sanity_check = sanity_check[sd == 0 & mean == 0]
cols_to_keep = colnames[!colnames %in% sanity_check$factor]

new_dt = new_dt[, cols_to_keep, with = FALSE]

rm(pca_results, pca_weights); gc()
new_train  = cbind(train[, 1:3], 
                   new_dt[1:nrow(train), 1:length(cols_to_keep)])
test_start = nrow(train)+1
test_end   = nrow(test)+nrow(train)
new_test   = cbind(test[, 1], 
                   new_dt[test_start:test_end, 1:length(cols_to_keep)])

rm(test_start, test_end, n_cols, n_facs, colnames, new_dt, sanity_check, cols_to_keep)
save(new_test, new_train, file = "PCA_v3A.RData")

# PCA for Linear Regression ----
pca_results = psych::principal(r = dt_cor, nfactors = sp_fac, rotate = "varimax", scores = TRUE, n.obs = nrow(dt_pca))
pca_weights = as.matrix(pca_results$loadings)
class(pca_weights) = "matrix"
pca_weights = as.data.table(pca_weights)
pca_weights = sapply(pca_weights, FUN = function(x) ifelse(x < 0.3 & x > -0.3, 0, x), simplify = TRUE)

# Now let's construct new data.tables:
new_dt     = as.matrix(dt_pca) %*% pca_weights
colnames   = as.character(seq(1, sp_fac, 1))
colnames   = paste0("factor", colnames)
new_dt     = as.data.table(new_dt)
new_dt     = `colnames<-`(new_dt, colnames)

rm(dt_cor, dt_eig, dt_eiv, pca_results, pca_weights); gc()
new_train  = cbind(train[, 1:3], 
                   new_dt[1:nrow(train), 1:length(colnames)])
test_start = nrow(train)+1
test_end   = nrow(test)+nrow(train)
new_test   = cbind(test[, 1], 
                   new_dt[test_start:test_end, 1:length(colnames)])

rm(test_start, test_end, colnames, dt_pca, test, train, new_dt)
save(new_test, new_train, file = "PCA_v3B.RData")