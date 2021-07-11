library(MASS)
library(rpart)
library(boot)

data(Boston)

set.seed(1234)

# integrating the boostrapping algorithm with rpart
bag_fn <- function(data, R = 100){
  MSE <- matrix(-99, R, 1)
  for(i in 1:R){
    ss <- sample(nrow(data), 200, replace = TRUE)
    sam <- data[ss,]
    test <- data[-ss,]
    cart_mod <- rpart(medv ~., sam)
    pred <- predict(cart_mod, test)
    MSE[i] <- mean((test$medv - pred)^2)
  }
  return(list("Mean SE" = mean(MSE),
              "Lower CI" = quantile(MSE, p = 0.05),
              "Upper CI" = quantile(MSE, p = 0.95)))
}


bag_fn(Boston)

# The MSE of this model is 27.04, which is slightly worse than the MSE of the pure (pruned) CART model (25.7).

cart_mod2 <- rpart(medv ~., data = Boston)
pruned_mod2 <- prune(cart_mod2, cp = 0.051)
summary(pruned_mod2)