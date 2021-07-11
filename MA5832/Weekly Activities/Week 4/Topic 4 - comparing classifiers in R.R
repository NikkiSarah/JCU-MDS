set.seed(2020)

# generate the dataset with n = 1000 and p = 2, such that the observations belong to two classes with a quadratic
#  decision boundary
x1 <- runif(1000) -0.5
x2 <- runif(1000) -0.5
y <- 1*(x1^2 - x2^2 > 0)

train <- data.frame(x1, x2, y)

# plot observations, coloured according to their class labels
plot(x1, x2, col = as.factor(y))

# fit a logistic regression model
log_mod <- glm(y ~ x1 + x2, data = train, family = "binomial")
summary(log_mod)

# obtained predicted class labels for all observations and plot the observations, coloured according to their
#  predicted class labels. The decision boundary should be linear
pred <- predict(log_mod, data = train, type = "response")
class_pred <- ifelse(pred > 0.5, 1, 0)

plot(x1, x2, col = as.factor(class_pred))

# fit a logistic regression model with non-linear functions of x1 and x2
log_mod2 <- glm(y ~ I(x1^2) + I(x2^2), data = train, family = "binomial")
summary(log_mod2)

# obtained predicted class labels for all observations and plot the observations, coloured according to their
#  predicted class labels. The decision boundary should be non-linear
pred <- predict(log_mod2, data = train, type = "response")
class_pred <- ifelse(pred > 0.5, 1, 0)

plot(x1, x2, col = as.factor(class_pred))

