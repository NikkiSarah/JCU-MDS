library(ElemStatLearn)

## Step 1: Import and scale the data
data(zip.train)
data(zip.test)

train <- as.data.frame(zip.train[,])
test <- as.data.frame(zip.test[,])

## Step 2: Data pre-processing
# scaling the data
normalise <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
train_norm <- as.data.frame(lapply(train[,2:ncol(train)], normalise))
train_norm$V1 <- train$V1

summary(train_norm[1:10])
summary(train_norm[257])

test_norm <- as.data.frame(lapply(test[,2:ncol(test)], normalise))
test_norm$V1 <- test$V1

# one-hot coding the outcome vector
library(nnet)

train_norm2 <- cbind(train_norm[, 1:256], class.ind(as.factor(train_norm$V1)))
names(train_norm2) <- c(names(train_norm)[1:256],
                        "d1","d2","d3","d4","d5","d6","d7","d8","d9","d0")

test_norm2 <- cbind(test_norm[, 1:256], class.ind(as.factor(test_norm$V1)))
names(test_norm2) <- c(names(test_norm)[1:256],
                       "d1","d2","d3","d4","d5","d6","d7","d8","d9","d0")

## Step 2: Classify data
# a) using a multinomial regression model (NN with no hidden layer)
library(neuralnet)

nn_mlr <- neuralnet(d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d0 ~.,
                    data = train_norm2,
                    hidden = 0,
                    linear.output = FALSE)

nn_mlr$result.matrix[1]

plot(nn_mlr)

# b) using a vanilla neural network (1 hidden layer, 12 hidden units)
vnn <- neuralnet(d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d0 ~.,
                 data = train_norm2,
                 hidden = 12,
                 linear.output = FALSE)

vnn$result.matrix[1]

plot(vnn)

# c) using a locally connected neural network

## Step 3: Comparing the training data results
train_actuals <- max.col(train_norm2[,257:266])

fitted_nn_mlr <- max.col(nn_mlr$response)
mean(fitted_nn_mlr == train_actuals)

fitted_vnn <- max.col(vnn$response)
mean(fitted_vnn == train_actuals)

# Training error for both models was 100%, but this was because the models were overfitting due to the presence of
#  more parameters than training observations.

## Step 4: Comparing the testing data results
test_actuals <- max.col(test_norm2[,257:266])

nn_mlr_pred <- predict(nn_mlr, newdata = test_norm2, type = "class")
mlr_pred_digit <- max.col(nn_mlr_pred)
mean(mlr_pred_digit == test_actuals)

vnn_pred <- predict(vnn, newdata = test_norm2, type = "class")
mlr_pred_digit <- max.col(vnn_pred)
mean(mlr_pred_digit == test_actuals)













## Step 2: Classify data using a multinomial regression model (NN with no hidden layer)
library(neuralnet)

set.seed(2020)
nn_mlr <- neuralnet(V1 ~.,
                    data = train,
                    hidden = 0,    # vector specifying the number of hidden neurons in each layer
                    linear.output = FALSE)

plot(nn_mlr)


## Step 3: Classify data using a NN, assuming a non-linear relationship between the output and features.
library(neuralnet)
# neural network with hidden layer c(1,1)
set.seed(2020)
nn1 <- neuralnet(X1 ~.,
                 data = train,
                 hidden = c(1,1),    # vector specifying the number of hidden neurons in each layer
                 linear.output = FALSE)

# neural network with hidden layer c(2,1)
set.seed(2020)
nn2 <- neuralnet(X1 ~.,
                 data = train,
                 hidden = c(2,1),
                 linear.output = FALSE)

# Step 4: Model results and classification error
nn1$result.matrix
nn1$result.matrix[1]

plot(nn1)

nn2$result.matrix
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

MC_rate2 <- (cm2[1,1] + cm2[2,2])/nrow(test)
MC_rate2

# The nn with the extra neuron in the second hidden layer is a little more accurate than the one with only a
#  single hidden layer.