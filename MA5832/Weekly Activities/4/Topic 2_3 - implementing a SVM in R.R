library(e1071)
library(ISLR)
library(caret)

data(Auto)

Auto$mpg_bn <- as.factor(ifelse(Auto$mpg > median(Auto$mpg), 1, 0))

set.seed(2020)
svm_mod <- svm(mpg_bn ~., data = Auto, kernel = "linear", cost = 1)
summary(svm_mod)

pred <- fitted(svm_mod)
confusionMatrix(pred, Auto$mpg_bn)

tune_out <- tune(svm, mpg_bn ~., data = Auto, kernel = "linear",
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
summary(tune_out)
# the best model is one in which cost = 1

#alternative tuning method
tune_out2 <- tune(svm, mpg_bn ~., data = Auto, kernel = "linear",
                  ranges = list(cost = seq(from = 0.001, to = 10, by = 0.01)))
summary(tune_out2)

final_mod <- tune_out2$best.model
summary(final_mod)

pred2 <- fitted(final_mod)
confusionMatrix(pred2, Auto$mpg_bn)
