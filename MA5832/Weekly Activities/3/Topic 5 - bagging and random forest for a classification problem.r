## Exercise 1: using randomForest for bagging
library(faraway)
library(randomForest)

data(kanga)

set.seed(1234)
bag_Ctree <- randomForest(sex ~., data = kanga, mtry = 15, importance = TRUE, na.action = na.omit)
bag_Ctree
# missing values option 1: exclude rows
# OOB estimate of the misclassification rate is 28.7% and the correct classification rate of the observed data is 71.3%

kanga_impute <- kanga
for(i in 2:ncol(kanga)){
  kanga_impute[,i] <- na.roughfix(kanga[,i])
}

set.seed(1234)
bag_Ctree2 <- randomForest(sex ~., data = kanga_impute, mtry = 15, importance = TRUE)
bag_Ctree2
# missing values option 2: imputation using median/mode
# OOB estimate of the misclassification rate is 24.3% and the correct classification rate of the observed data is 75.7%
#  imputation using the median/mode offers better performance than excluding the rows

set.seed(1234)
kanga_impute2 <- rfImpute(sex ~., kanga)
bag_Ctree3 <- randomForest(sex ~., data = kanga_impute2, mtry = 15, importance = TRUE)
bag_Ctree3
# missing values option 3: imputation using the proximity matrix
# OOB estimate of the misclassification rate is 23.7% and the correct classification rate of the observed data is 76.4%
#  imputation using the proximity matrix performs slightly better than using the median/mode

imp <- data.frame(importance(bag_Ctree3, type = 1))
par(mfrow = c(2,1), cex = 0.4)
barplot(sort(bag_Ctree3$importance[,3], decreasing = TRUE))
barplot(sort(bag_Ctree3$importance[,4], decreasing = TRUE))
# both charts agree that basilar length is the most important variable in predicting kangaroo gender, but there is a
#  disagreement in the second-most important variable: palate length has the second highest reduction in the
#  difference between observed and permuted OOB samples whilst ramus height is the second-best predictor in reducing
#  node impurity when used for splitting


## Exercise 2: applying randomForest to a classification problem
library(faraway)
library(randomForest)

data(kanga)

mtry = ncol(kanga)/3

set.seed(1234)
Ctree <- randomForest(sex ~., data = kanga, mtry = mtry, importance = TRUE, na.action = na.omit)
Ctree
# missing values option 1: exclude rows
# OOB estimate of the misclassification rate is 27.7% and the correct classification rate of the observed data is 72.3%

kanga_impute <- kanga
for(i in 2:ncol(kanga)){
  kanga_impute[,i] <- na.roughfix(kanga[,i])
}

set.seed(1234)
Ctree2 <- randomForest(sex ~., data = kanga_impute, mtry = mtry, importance = TRUE)
Ctree2
# missing values option 2: imputation using median/mode
# OOB estimate of the misclassification rate is 24.3% and the correct classification rate of the observed data is 75.7%
#  imputation using the median/mode offers better performance than excluding the rows

set.seed(1234)
kanga_impute2 <- rfImpute(sex ~., kanga)
Ctree3 <- randomForest(sex ~., data = kanga_impute2, mtry = mtry, importance = TRUE)
Ctree3
# missing values option 3: imputation using the proximity matrix
# OOB estimate of the misclassification rate is 23.7% and the correct classification rate of the observed data is 76.4%
#  imputation using the proximity matrix performs slightly better than using the median/mode

imp <- data.frame(importance(Ctree3, type = 1))
par(mfrow = c(2,1), cex = 0.4)
barplot(sort(Ctree3$importance[,3], decreasing = TRUE))
barplot(sort(Ctree3$importance[,4], decreasing = TRUE))
# both charts agree that basilar length is the most important variable in predicting kangaroo gender, but there is a
#  disagreement in the second-most important variable: zygomatic width has the second highest reduction in the
#  difference between observed and permuted OOB samples whilst ramus height is the second-best predictor in reducing
#  node impurity when used for splitting

# Question 1: There was just about no difference between the bagged and randomForest trees for this dataset.
# Question 2: The most important variables stayed the same except for the second-most important for the reduction
#  in the difference between observed and permuted OOB samples