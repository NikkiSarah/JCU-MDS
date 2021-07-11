## Exercise 1: Regression trees in R
library(MASS)
library(rpart)

data(Boston)

set.seed(1234)
M1 <- rpart(medv ~., data = Boston)
summary(M1)

# plot the tree
par(xpd = TRUE) # all plotting clipped to the figure region
plot(M1, compress = TRUE) # routine attempts a more compact arrangement of the tree
text(M1, digits = 2, use.n = TRUE) # adds size of leaf/terminal node to plot below the mean

# plot the complexity parameter
plotcp(M1)

# plot suggests that cp should be set to 0.051 as the errors remain largely unchanged at this value.
# the rpart manual suggests that a good choice for alpa is the left-most value for which the mean of the error is below
#  the dotted line.

# prune the original tree and re-plot
M2 <- prune(M1, cp = 0.051)
par(xpd = TRUE)
plot(M2, compress = TRUE)
text(M2, digits = 2, use.n = TRUE)

# after pruning, the only variables used in the prediction of housing price were 'rms' and 'lstat'.
# the highest housing price occurs when rm > 7.44 with a predicted median value of $45,000.
# when 6.94 < rm < 7.44, the median house price falls to $32,000
# when rm < 6.94, then lstat impacts on house prices. If lstat >= 14.4 then the median house price is $23,000.

summary(M2)

# explore important predictors for housing price
barplot(M2$variable.importance, cex.names = 0.8)

# this confirms that the most important variables are 'rm' and 'lstat'.

# test goodness-of-fit by examining the residuals
par(mfrow = c(1,2))
plot(predict(M2), residuals(M2))
qqnorm(residuals(M2))
qqline(residuals(M2))

# residual variance is reasonably constant among the fitted values (LHS plot) but there are some outliers in the
#  data (RHS plot)

MSE <- mean(residuals(M2)^2)
# regression tree MSE is 25.7

# use rattle to plot a better plot
library(rattle)
library(rpart.plot)
fancyRpartPlot(M2)


## Exercise 2: Creating a regression tree
library(ISLR)
library(rpart)
library(rattle)
library(rpart.plot)
library(tidyverse)

data(Hitters)

set.seed(1234)
M1 <- rpart(Salary ~., data = Hitters)
summary(M1)

fancyRpartPlot(M1)
plotcp(M1)
# plot suggests that cp should be set to 0.21 as the errors remain largely unchanged at this value.
# the rpart manual suggests that a good choice for alpa is the left-most value for which the mean of the error is
#  below the dotted line.

# prune the original tree and re-plot
M2 <- prune(M1, cp = 0.21)
fancyRpartPlot(M2)

# after pruning, the only variable used in the prediction of players' salary was 'CHits'
# salary was higher when CHits >= 450 with a predicted value of $783,000

summary(M2)

# explore important predictors for player's salary
barplot(M2$variable.importance, cex.names = 0.8)

# the most important predictor for player salary is CHits, followed by CAtBat, CRuns, CRBI, CWalks and Years.
#  However, the relative importance for the first six variables was very similar, which suggests the
#  predictors are correlated. This can be checked using Pearson correlation

d1 <- Hitters %>% 
  select("CHits", "CAtBat", "CRuns", "CRBI", "CWalks", "Years")
round(cor(d1, use = "pairwise.complete.obs"), 2)

# test goodness-of-fit by examining the residuals
par(mfrow = c(1,2))
plot(predict(M2), residuals(M2))
qqnorm(residuals(M2))
qqline(residuals(M2))

# residual variance increases with the value of predictors and there are outliers in the data

MSE <- mean(residuals(M1)^2)
# regression tree MSE is 69320
