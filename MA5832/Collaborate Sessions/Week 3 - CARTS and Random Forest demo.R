library(rpart)
library(rpart.plot)

data(ptitanic)

set.seed(1234)

no_obs <- dim(ptitanic)[1]
test_index <- sample(no_obs, size = as.integer(no_obs*0.2), replace = FALSE) # 20% data records for test
training_index <- -test_index

training = ptitanic[training_index,]
testing = ptitanic[test_index,]

set.seed(1234)
tree <- rpart(survived~., data=training)

rpart.plot(tree)
rpart.rules(tree)

result <- predict(tree, testing, type ="class")
(t <- table(testing[,2],result))

(accuracy = sum(diag(t)) / sum((t)))


set.seed(1234)
library(randomForest)
r_tree <- randomForest(survived ~., data=training, mtry=2, importance=TRUE, na.action=na.roughfix)
r_result <- predict(r_tree, testing)

(t <- table(testing[,2],r_result))

(accuracy = sum(diag(t)) / sum((t)))