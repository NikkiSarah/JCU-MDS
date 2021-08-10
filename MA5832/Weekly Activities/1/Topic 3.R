setwd("~/Training & Study/JCU/2020_Data Mining and Machine Learning/Week 1")

library(glmpath)
library(tidyverse)
library(caret)

data(heart.data)
data <- cbind(heart.data$x, heart.data$y)
data_df <- as.data.frame(data)

names(data_df)[names(data_df) == "V10"] <- "disease"

set.seed(123)
train_samples <- data_df$disease %>% 
  createDataPartition(p = 0.7, list = FALSE)
train_data <- data_df[train_samples, ]
test_data <- data_df[-train_samples, ]

preproc_param <- train_data %>% 
  preProcess(method = c("center", "scale"))
norm_train <- preproc_param %>% 
  predict(train_data)
norm_test <- preproc_param %>% 
  predict(test_data)

library(MASS)
mod <- lda(disease ~., data = norm_train)
preds <- mod %>% predict(norm_test)
mean(preds$class == norm_test$disease)
