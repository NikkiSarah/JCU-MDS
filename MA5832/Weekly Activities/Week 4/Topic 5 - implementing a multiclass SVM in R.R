library(ISLR)
library(e1071)

data(Khan)

train <- cbind(Khan$xtrain, Khan$ytrain)
test <- cbind(Khan$xtest, Khan$ytest)
train <- data.frame(train)
test <- data.frame(test)

train <- rename(train, y = X2309)
train$y <- as.factor(train$y)
test <- rename(test, y = X2309)
test$y <- as.factor(test$y)

dim(train)
dim(test)

# fitting the multi-class maximal hyperplane linear classifier
linear_mod <- svm(y ~., data = train, kernel = "linear", cost = 1)
summary(linear_mod)

pred <- predict(linear_mod, data = train)
confusionMatrix(pred, train$y)

# tuning model by adding a classification error
tune_out <- tune(svm, y ~., data = train,
                 ranges = list(cost = c(0.001:20)))
summary(tune_out)

final_linear_mod <- tune_out$best.model
summary(final_linear_mod)

# fitting a quadratic classifier
quad_mod <- svm(y ~., data = train, kernel = "polynomial", degree = 2)
summary(quad_mod)

pred <- predict(quad_mod, data = train)
confusionMatrix(pred, train$y)

# comparing the models
linear_pred <- predict(final_linear_mod, newdata = test)
confusionMatrix(linear_pred, test$y)

quad_pred <- predict(quad_mod, newdata = test)
confusionMatrix(quad_pred, test$y)

# on the training data, both models had perfect accuracy, but on the test data the quadratic model only had 55%
#  accuracy, which was mainly misclassifying observations from groups 2 and 3 as 4. The linear model in comparison
#  still had a 90% accuracy rate, due to misclassifying two observations from group 3 as group 2.
