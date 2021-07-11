setwd("~/Training & Study/JCU/2020_Data Mining and Machine Learning/Week 1")

# Q1
library(glmpath)

data(heart.data)
data <- cbind(heart.data$x, heart.data$y)
View(data)

data_df <- as.data.frame(data)
View(data_df)

names(data_df)[names(data_df) == "V10"] <- "disease"

# Q2
set.seed(20)
kclust <- kmeans(data_df[,c(1,3,5,7)], 2)

# Q3
data_df$disease_clust <- kclust$cluster

# Q4
str(kclust)
kclust

library(factoextra)
fviz_cluster(kclust, data = data_df)

# Q5
library(tidyverse)
ggplot(data_df, aes(x = sbp, y = ldl,
                    colour = as.factor(disease_clust),
                    shape = as.factor(disease))) +
  geom_point() +
  scale_colour_discrete(name = "Cluster") +
  scale_shape_discrete(name = "Outcome") +
  xlab("Systolic blood pressure") +
  ylab("Low density lipoprotein cholesterol")

# Q6
data_df2 <- subset(data_df, select = c(sbp, ldl, obesity, famhist, disease))
View(data_df2)

mod = glm(disease ~ ., data = data_df2, family = binomial)
summary(mod)

# Q7
probabilities <- mod %>% predict(data_df2, type = "response")
pred_class <- ifelse(probabilities > 0.5, 0, 1)

disease_pred <- data.frame(cbind(data_df$disease, data_df$disease_clust, pred_class))
disease_pred <- disease_pred %>% mutate_if(is.numeric, as.factor)

names(disease_pred)[names(disease_pred) == "V1"] <- "disease_outcome"
names(disease_pred)[names(disease_pred) == "V2"] <- "cluster_pred"
names(disease_pred)[names(disease_pred) == "pred_class"] <- "logreg_pred"

disease_pred$cluster_pred <- plyr::mapvalues(disease_pred$cluster_pred, from = c("1", "2"), to = c("0", "1"))

# Q8
table(disease_pred$disease_outcome, disease_pred$logreg_pred)
mean(pred_class == data_df$disease)

# Q9
table(disease_pred$logreg_pred, disease_pred$cluster_pred)