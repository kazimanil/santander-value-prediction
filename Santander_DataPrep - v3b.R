summary =
  data.table( colname    = colnames(train[, 4:ncol(train)]),
              train_sd   = as.data.table(sapply(train[, 4:ncol(train)], sd, na.rm = TRUE))$V1,
              train_mean = as.data.table(sapply(train[, 4:ncol(train)], mean, na.rm = TRUE))$V1,
              train_rows = nrow(train),
              test_sd    = as.data.table(sapply(test[, 2:ncol(test)], sd, na.rm = TRUE))$V1,
              test_mean  = as.data.table(sapply(test[, 2:ncol(test)], mean, na.rm = TRUE))$V1,
              test_rows  = nrow(test))
summary[, t_value := (train_mean - test_mean) / sqrt((train_sd^2/train_rows) + (test_sd^2 /test_rows))]
summary[, t_pos := qt(p = 0.99, df = nrow(test) + nrow(train) - 2)]
summary[, t_neg := -qt(p = 0.99, df = nrow(test) + nrow(train) - 2)]
selected_cols = summary[t_value >= t_neg & t_value <= t_pos]$colname
train = train[, append("ID", append("target", append("target_boxcox", selected_cols))), with = FALSE]
test  = test[, append("ID",selected_cols), with = FALSE]
rm(summary, selected_cols)

dt_pca = rbind(train[, 4:ncol(train)], test[, 2:ncol(test)])
n_cols = ncol(dt_pca)  # Number of columns
dt_cor = cor(dt_pca)   # Correlation Matrix
dt_eig = eigen(dt_cor) # EigenValues & EigenVectors
dt_eiv = dt_eig$values # EigenValues
n_facs = length(dt_eiv[dt_eiv > 1]) # Amount of Factors to be derived based on "EigenValue > 1" criterion.
sp_fac = 11 # Amount of Factors to be derived based on "ScreePlot" criterion.

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

rm(pca_results, pca_weights, cols_to_keep); gc()
new_train  = cbind(train[, 1:3], 
                   new_dt[1:nrow(train)])
test_start = nrow(train)+1
test_end   = nrow(test)+nrow(train)
new_test   = cbind(test[, 1], 
                   new_dt[test_start:test_end])

rm(test_start, test_end, n_cols, n_facs, colnames, new_dt, sanity_check)
saveRDS(new_test, file = "test_PCAv4_xgb.RDS")
saveRDS(new_train, file = "train_PCAv4_xgb.RDS")

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
                   new_dt[1:nrow(train)])
test_start = nrow(train)+1
test_end   = nrow(test)+nrow(train)
new_test   = cbind(test[, 1], 
                   new_dt[test_start:test_end,])

rm(test_start, test_end, colnames, dt_pca, test, train, new_dt, sp_fac)
saveRDS(new_test, file = "test_PCAv4_linreg.RDS")
saveRDS(new_train, file = "train_PCAv4_linreg.RDS")
