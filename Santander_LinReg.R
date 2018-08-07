# Santander Linear Regression Estimation ----
lambda = 0.12
test  = readRDS(file = "test_PCAv3_linreg.RDS")
train = readRDS(file = "train_PCAv3_linreg.RDS")

# There is enough evidence to suspicious about factors not being normally-distributed. Let's have box-cox transformation over them as well.
for(i in 1:4){
  tr_i = 3 + i; ts_i = 1 + i;
  target = train[, tr_i, with = FALSE] + 1
  colnames(target) = c("target")
  box = boxcox(target$target ~ 1,
               lambda = seq(-1, 1, 0.01),
               plotit = FALSE)
  cox = data.table(lambda = box$x, 
                   logLL  = box$y)[order(-logLL)]
  lambda_x = cox[1]$lambda # to seperate from original lambda.
  newcol = paste0("factor_bc_", i)
  newval = (target$target ^ lambda_x - 1) / lambda_x
  train[, (newcol) := newval]
  
  target2 = test[, ts_i, with = FALSE] +1 
  colnames(target2) = c("target")
  newval2 = (target2$target ^ lambda_x - 1) / lambda_x
  test[, (newcol) := newval2]
}
rm(box, cox, lambda_x, i, tr_i , ts_i, newcol, newval, newval2, target, target2)

linreg = step(object = lm(data = train, formula = target_boxcox ~ factor_bc_1 + factor_bc_2 + factor_bc_3 + factor_bc_4),
              direction = "backward", trace = 0)
test[, target_boxcox := predict(linreg, newdata = test[, 6:8])]
test[, target := (target_boxcox * lambda + 1) ^ ( 1 / lambda)]
fwrite(test[, c(1,11)], file = "submission_linreg.csv")
