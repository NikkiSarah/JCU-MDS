library(randomForest)
library(MASS)

data(Boston)

## Exercise 1: using randomForest for bagging
set.seed(1234)
bag_Rtree <- randomForest(medv ~., data = Boston, mtry = 13, importance = TRUE)
bag_Rtree

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
# 'lstat' has the largest reduction in the difference between observed and permutated OOB samples (top plot)
# 'rm' is the best predictor in reducing node impurity while used for splitting


## Exercise 2: applying random forest to a regression problem
mtry = ncol(Boston)/3
set.seed(1234)
Rtree <- randomForest(medv ~., data = Boston, mtry = mtry, importance = TRUE)
Rtree

residuals <- Boston$medv - Rtree$predicted
MSE_Rtree <- mean(residuals^2)
# MSE is 9.67, which is slightly lower than that of a bagged tree

plot(Boston$medv, residuals)
# the model has reasonable predictions for housing values less than $50,000, but it overestimates housing with values
#  above $50,000

Rtree$importance
par(mfrow = c(2,1), cex = 0.7)
barplot(sort(Rtree$importance[,1], decreasing = TRUE))
barplot(sort(Rtree$importance[,2], decreasing = TRUE))