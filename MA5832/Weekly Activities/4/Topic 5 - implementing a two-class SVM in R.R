library(e1071)
library(glmpath)
library(tidyverse)
library(caret)

data(heart.data)
heart <- cbind(heart.data$x, heart.data$y)
heart <- data.frame(heart)

heart <- rename(heart, outcome = V10)
heart$outcome <- as.factor(heart$outcome)

set.seed(2020)
train_ind <- sample(1:nrow(heart), 0.7*nrow(heart))

train <- heart[train_ind,]
test <- heart[-train_ind,]

svm_mod <- svm(outcome ~., data = train, cost = 1, gamma = (1/9))
summary(svm_mod)

pred <- predict(svm_mod, data = train)
confusionMatrix(pred, train$outcome)

tune_out <- tune(svm, outcome ~., data = train,
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100, 150, 200),
                               gamma = c(0.001, 0.01, 0.1, 1, 5, 10, 100, 150, 200)))
summary(tune_out)

final_mod <- tune_out$best.model
summary(final_mod)

pred2 <- predict(final_mod, newdata = test)
confusionMatrix(pred2, test$outcome)