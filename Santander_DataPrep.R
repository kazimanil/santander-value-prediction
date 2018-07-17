# Data Input ----
train = fread("train.csv")
test  = fread("test.csv")
sample = fread("sample_submission.csv")

library("xgboost")
# Exclude Variables which Test & Train distribute differently.
summary =
	data.table(colname    = colnames(train[, 3:ncol(train)]),
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
train = train[, selected_cols, with = FALSE]
test  = test[, selected_cols, with = FALSE]
rm(summary, selected_cols)
