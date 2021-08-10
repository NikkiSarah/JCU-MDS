library(ISLR)
library(supervisedPRIM)

data(Caravan)

x <- Caravan[, 1:85]
y <- factor(Caravan$Purchase)

prim_mod <- supervisedPRIM(x,y,threshold.type = 1)
summary(prim_mod, print.box = TRUE)

# There were no missing values so the full dataset could be used for the algorithm.
# The algorithm found that one box has a high concentration of customers that purchased a caravan insurance policy.
#  That is, 68.7% of all the data fell in Box 1.

pred <- predict(prim_mod,x,y.fun.flag = TRUE)
table(pred, y)

(3858+207)/nrow(Caravan)

# The table of predicted vs actual outcomes indicated that 69.8% of customer purchase decisions were correctly
#  predicted by the alogorithm