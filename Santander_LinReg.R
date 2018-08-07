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

# Now selected variables will be put into PCA.