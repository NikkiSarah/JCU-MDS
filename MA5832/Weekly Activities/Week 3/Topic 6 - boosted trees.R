library(MASS)
library(gbm)

data(Boston)

set.seed(1)

train_data <- sample(nrow(Boston), nrow(Boston)/2)
train <- Boston[train_data,]
test <- Boston[-train_data,]
medv_test <- Boston[-train_data, "medv"]

boost_tree <- gbm(medv ~., data = train, distribution = "gaussian", n.trees = 5000, interaction.depth = 4)

summary(boost_tree)
# lstat and rm are by far the most important variables

par(mfrow = c(1,2))
plot(boost_tree, i = "lstat")
plot(boost_tree, i = "rm")
# these partial dependence plots indicate that median house prices decrease as lstat increases and increase as rm
#  increases

boost_yhat <- predict(boost_tree, newdata = test, n.trees = 5000)
boost_MSE <- mean((boost_yhat - medv_test)^2)
# the test MSE is 19.4, which is similar to that of the random forest (18.1) and far superior to that of a bagged tree
#  (23.6)



## For comparison
library(randomForest)
# Bagged tree
set.seed(1)
mtry = ncol(Boston) - 1

bag_tree <- randomForest(medv ~., data = train, mtry = mtry, importance = TRUE)
bag_tree

bag_tree$importance
par(mfrow = c(2,1), cex = 0.7)
barplot(sort(bag_tree$importance[,1], decreasing = TRUE))
barplot(sort(bag_tree$importance[,2], decreasing = TRUE))
# like the boosted tree, lstat and rm are by far the most important variables

bag_yhat <- predict(bag_tree, newdata = test)
bag_MSE <- mean((bag_yhat - medv_test)^2)
# the test MSE is 23.4

# Random Forest tree
set.seed(1)
mtry = round((ncol(Boston) - 1)/3)

rf_tree <- randomForest(medv ~., data = train, mtry = mtry, importance = TRUE)
rf_tree

rf_tree$importance
par(mfrow = c(2,1), cex = 0.7)
barplot(sort(rf_tree$importance[,1], decreasing = TRUE))
barplot(sort(rf_tree$importance[,2], decreasing = TRUE))
# like the bagged tree, lstat and rm are by far the most important variables

rf_yhat <- predict(rf_tree, newdata = test)
rf_MSE <- mean((rf_yhat - medv_test)^2)
# the test MSE is 18.1





residuals <- Boston$medv - bag_Rtree$predicted
MSE_bagRtree <- mean(residuals^2)
# MSE is 10.32, which is much lower than that of a simple CART regression tree

plot(Boston$medv, residuals)
# the model has reasonable predictions for housing values less than $50,000, but it overestimates housing with values
#  above $50,000

bag_Rtree$importance
par(mfrow = c(2,1), cex = 0.7)
barplot(sort(bag_Rtree$importance[,1], decreasing = TRUE))
barplot(sort(bag_Rtree$importance[,2], decreasing = TRUE))

