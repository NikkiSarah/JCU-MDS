library(tidyverse)
library(readxl)

dividend <- read_excel("dividends.xlsx")
str(dividend)

## Step 1: Normalise/Standardise the data
# scale method
dividend_norm <- as.data.frame(scale(dividend[,2:ncol(dividend)]))
dividend_norm$dividend <- dividend$dividend

# min-max method (same as using multiClust::nor.min.max())
normalise <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
dividend_norm2 <- as.data.frame(lapply(dividend[,2:ncol(dividend)], normalise))
dividend_norm2$dividend <- dividend$dividend

summary(dividend_norm)
summary(dividend_norm2)

## Step 2: Split into training and test sets
set.seed(2020)
train_ind <- sample(nrow(dividend_norm2), nrow(dividend_norm2)*0.7)
train <- dividend_norm2[train_ind, ]
test <- dividend_norm2[-train_ind, ]

## Step 3: Classify data using a NN, assuming a non-linear relationship between the output and features.
library(neuralnet)
# neural network with hidden layer c(1,1)
set.seed(2021)
nn1 <- neuralnet(dividend ~.,
                 data = train,
                 hidden = c(1,1),    # vector specifying the number of hidden neurons in each layer
                 linear.output = FALSE)
nn1$result.matrix

nn1_pred <- predict(nn1, newdata = test, type = "class")
nn1_class <- ifelse(nn1_pred > 0.5, 1, 0)
table(predicted = nn1_class, actual = test$dividend)

plot(nn1)

# neural network with hidden layer c(2,1)
set.seed(2022)
nn2 <- neuralnet(dividend ~.,
                 data = train,
                 hidden = c(2,1),
                 linear.output = FALSE)
nn2$result.matrix

nn2_pred <- predict(nn2, newdata = test, type = "class")
nn2_class <- ifelse(nn2_pred > 0.5, 1, 0)
table(predicted = nn2_class, actual = test$dividend)

plot(nn2)

# The nn with the extra neuron in the second hidden layer is a little more accurate than the one with only a
#  single hidden layer.