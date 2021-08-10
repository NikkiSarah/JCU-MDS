## Task: Use random forest and bagging to develop models that identify which 'new_clients' customers the direct marketing
##        campaign should target based on the 'bank' dataset.
library(randomForest)
library(tidyverse)

# Part 1: Determine the 'best' model

# Step 1: Import data
# Categorical variables imported as factors
bank <- read.csv2("bank.csv", stringsAsFactors = TRUE)
new_clients <- read.csv2("new_clients.csv", sep = ",", stringsAsFactors = TRUE)

str(bank)
str(new_clients)

## Step 2: Check for missing values
nrow(bank[rowSums(is.na(bank)) >= 1,])
apply(bank, 2, function(x) sum(is.na(x)))

# No missing values officially but 'unknowns' could be considered such. They will be ignored for now.

# Step 3: Training the models
# Model 1: Bagged tree
mtry = ncol(bank) - 1

set.seed(1234)
bag_Ctree <- randomForest(y ~., data = bank, mtry = mtry, importance = TRUE)
bag_Ctree

# OOB estimate of the misclassification rate is 10.04% and
#  the correct classification rate of the observed data is 90.0%

imp <- data.frame(importance(bag_Ctree, type = 1))
par(mfrow = c(2,1), cex = 0.4)
barplot(sort(bag_Ctree$importance[,3], decreasing = TRUE))
barplot(sort(bag_Ctree$importance[,4], decreasing = TRUE))

# Both charts agree that the top three predictors for the outcome of the marketing campaign are duration, month and
#  day.

# Model 2: Random forest tree
mtry = sqrt(ncol(bank)-1)

set.seed(1234)
Ctree <- randomForest(y ~., data = bank, mtry = mtry, importance = TRUE)
Ctree

# OOB estimate of the misclassification rate is 9.69% and
#  the correct classification rate of the observed data is 90.3%

imp <- data.frame(importance(Ctree, type = 1))
par(mfrow = c(2,1), cex = 0.4)
barplot(sort(Ctree$importance[,3], decreasing = TRUE))
barplot(sort(Ctree$importance[,4], decreasing = TRUE))

# In the random forest case, the charts disagree on the most important predictors. Month, duration and contact are the
#  top three predictors when considering the reduction in the difference between observed and permuted OOB samples
#  whereas duration, month and balance are the top three predictors in reducing node impurity when used for
#  splitting.

# Step 4: Prediction using the test set
bag_pred <- predict(bag_Ctree, newdata = new_clients, type = "response")
summary(bag_pred)

rf_pred <- predict(Ctree, newdata = new_clients, type = "response")
summary(rf_pred)

# Prediction on the test set indicates that only 97 of the new customers (9.7%) should be targeted with the marketing
#  campaign.

# Prediction on the test set with the bagged model indicates that only 105 of the new customers (10.5%) should be
#  targeted with the new campaign. The random forest produces similar results (97 or 9.7%).

table(bag_pred, rf_pred)

## Question 1: How do the results compare between RF and bagging?
#   The training data performance metrics for the bagged model is very similar to that for the random forest
#    model, noting that the 'unknowns' in the dataset were treated as a valid response and not as missing data.

## Question 2: What is the error rate of each model?
#   The OOB estimates of the misclassification rate for the bagged model was 10.04% compared to 9.699% for the random
#    forest model.

## Question 3: What are the key attributes?
#   Both the bagged and random forest models agreed that the two most important predictors were duration and month,
#    although not necessarily in the same order. The third most important predictor was contact for both bagged model
#    metrics, and either contact or balance for the random forest model.

## Question 4: What proportion of customers should be targeted according to each model?
#   Using the test set of new clients, the bagged model indicated that only 10.5% should be targeted with the new
#    campaign, which was very similar to the random forest's 9.7%.

## Question 5: Do your models agree on the people who should be targeted in this campaign?
#   The two models largely agree, but there are 20 customers that they don't agree on.