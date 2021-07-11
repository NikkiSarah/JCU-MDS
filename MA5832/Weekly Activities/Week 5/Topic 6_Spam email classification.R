spam <- read.csv("spambase.data")
str(spam)

## Step 1: Scale the data
normalise <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
spam_norm <- as.data.frame(lapply(spam[,2:ncol(spam)], normalise))
spam_norm$spam <- spam$spam

## Step 2: Split into training and test sets
set.seed(2020)
train_ind <- sample(nrow(spam_norm), nrow(spam_norm)*0.8)
train <- spam_norm[train_ind, ]
test <- spam_norm[-train_ind, ]

## Step 3: Classify data using a NN, assuming a non-linear relationship between the output and features
#   # nb incl of a hidden layer implies a non-linear relationship
library(neuralnet)
# neural network with a single hidden layer
set.seed(2021)
nn1 <- neuralnet(X1 ~.,
                 data = train,
                 hidden = 10,    # vector specifying the number of layers and hidden neurons in each layer.
                 linear.output = FALSE)  # indicates a classification problem (TRUE indicates regression)

# neural network with two hidden layers
set.seed(2022)
nn2 <- neuralnet(X1 ~.,
                 data = train,
                 hidden = c(10,5),
                 linear.output = FALSE)

# Step 4: Model results and classification error
#nn1$result.matrix # too big
nn1$result.matrix[1]

plot(nn1)

#nn2$result.matrix
nn2$result.matrix[1]

plot(nn2)

#Step 5: Apply model to the test data and report the model accuracy based on the confusion matrix
nn1_pred <- predict(nn1, newdata = test, type = "class")
nn1_class <- ifelse(nn1_pred > 0.5, 1, 0)
cm1 <- table(predicted = nn1_class, actual = test$X1)
cm1

MC_rate1 <- (cm1[1,1] + cm1[2,2])/nrow(test)
MC_rate1

nn2_pred <- predict(nn2, newdata = test, type = "class")
nn2_class <- ifelse(nn2_pred > 0.5, 1, 0)
cm2 <- table(predicted = nn2_class, actual = test$X1)
cm2

MC_rate2 <- (cm2[1,1] + cm2[2,2])/nrow(test)
MC_rate2

# The nn with the extra hidden layer actually does slightly worse than the first neural network (but both performed
#  very well)