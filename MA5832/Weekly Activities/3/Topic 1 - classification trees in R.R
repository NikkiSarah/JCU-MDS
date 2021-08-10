## Exercise 1: Classification trees in R
library(faraway)
library(tidyverse)
library(rpart)
library(rpart.plot)
library(rattle)

data(kanga)
names(kanga)

# generate a new column to combine species and sex
kanga1 <- kanga %>% unite("SS", c("species", "sex"))

# check for missing data
nrow(kanga1[rowSums(is.na(kanga1)) >= 1,])

## missing values by column name
apply(kanga1, 2, function(x) sum(is.na(x)))

# there are 47 records with at least one missing predictor, with most located in the 'palate.width', 'mandible.length'
#  or 'occipital.depth' variables
# there are many different methods for dealing with missing values and the default rpart approach will be used here
#  i.e. the non-missing parts of the records are used to inform splits and when a missing value is used for a primary
#   split, the record is excluded from the determination of the subsequent split
# the default splitting rule - the Gini index - is used to measure impurity

set.seed(1234)
M1 <- rpart(SS ~., data = kanga1)
fancyRpartPlot(M1)

plotcp(M1)

# alpha = 0.02 is the left-most value under the dotted line, so this is used to prune the tree
M2 <- prune(M1, cp = 0.02)
fancyRpartPlot(M2)

# determining the missclassification rate of the tree
pre <- predict(M2, kanga1, type = "class")
tab <- table(pre, kanga1$SS)

# the pruned tree correctly classified 61% of the observations, which was 1% lower than the full tree and indicates
#  that the full tree overfitted the data

# the pruned tree was reasonably accurate at predicting both sexes of 'Fuliginosus sp.', female 'Melanops sp.'
#  and male 'Giganateus sp.', but performed poorly at predicting female 'Giganateus sp.' and male 'Melanops sp.'.

barplot(M2$variable.importance, cex.names = 0.8, las = 2) # las creates vertical x-axis labels

# the most important predictor for kangaroo species is nasal length, followed by occipitonasal length and basilar
#  length. However, the relative importance for the first seven variables was very similar, which suggests the
#  predictors are correlated. This can be checked using Pearson correlation

d1 <- kanga1 %>% 
  select("nasal.length", "palate.length", "occipitonasal.length", "ramus.height", 
         "basilar.length", "zygomatic.width", "lacrymal.width")
round(cor(d1, use = "pairwise.complete.obs"), 2)


## Exercise 2: developing a model using a classification tree
library(tidyverse)
library(rpart)
library(rattle)
library(rpart.plot)

## Step 1: Import the data
bank <- read.csv2("bank.csv")

str(bank)
# replace 'unknown's with 'NA'
bank2 <- bank %>% 
  naniar::replace_with_na(replace = list(job = "unknown"))

bank2 <- bank2 %>% 
  naniar::replace_with_na(replace = list(education = "unknown"))

bank2 <- bank2 %>% 
  naniar::replace_with_na(replace = list(contact = "unknown"))

bank2 <- bank2 %>% 
  naniar::replace_with_na(replace = list(poutcome = "unknown"))

## Step 2: Check for missing values
nrow(bank2[rowSums(is.na(bank2)) >= 1,])
apply(bank2, 2, function(x) sum(is.na(x)))

# there are 3757 records with at least one missing predictor, with most located in the 'contact' and 'poutcome'
#  variables
# there are many different methods for dealing with missing values and the default rpart approach will be used here
#  i.e. the non-missing parts of the records are used to inform splits and when a missing value is used for a primary
#   split, the record is excluded from the determination of the subsequent split
# the default splitting rule - the Gini index - is used to measure impurity

set.seed(1234)
M1 <- rpart(y ~., data = bank2)
fancyRpartPlot(M1)

plotcp(M1)

# alpha = 0.026 is the left-most value under the dotted line, so this is used to prune the tree
M2 <- prune(M1, cp = 0.026)
fancyRpartPlot(M2)

# determining the missclassification rate of the tree
pre <- predict(M2, bank2, type = "class")
tab <- table(pre, bank2$y)

# the pruned tree correctly classified 90% of the observations, which was 1% lower than the full tree and indicates
#  that the full tree overfitted the data

# the pruned tree was much better at correctly classifying if the client had not subscribed to a term deposit than
#  if they had

barplot(M2$variable.importance, cex.names = 0.8, las = 2) # las creates vertical x-axis labels

# the most important predictor for the outcome of the marketing campaign was 'duration' followed by 'poutcome' with
#  'pdays', 'previous' and 'marital' rounding out the top five (albeit distantly)