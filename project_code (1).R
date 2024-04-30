library(caret)
library(rsample)
library(RWeka)
library(rpart)
library(rpart.plot)
library(MASS)
library(kernlab)
library(pROC)
library(dplyr)
library(C50)
library(e1071)
library(MASS)
library(ROSE)
library(FSelector)
library(Boruta)

# Load data
getwd()
setwd('Downloads')
df <- read.csv("/Users/jyoti/Downloads/project_dataset.csv")

# Check Dimensionality
dim(df)

# Check for missing values
sum(is.na(df))

# Check for duplicates
sum(duplicated(df))
df[duplicated(df) | duplicated(df, fromLast = TRUE), ]

# Remove duplicates
df_no_duplicated <- df[!duplicated(df),]
df_no_duplicated
sum(is.na(df_no_duplicated))

# Check outlier
boxplot(df)
boxplot(df_no_duplicated)
df3 <- df_no_duplicated
boxplot(df3, plot=FALSE)
Q1 <- quantile(df3$V2021, 0.25)
Q3 <- quantile(df3$o_bullied, 0.75)
IQR <- Q3 - Q1

# Check for duplicates in new dataset
sum(duplicated(df_no_duplicated))

# Remove features with zero variance
df_no_duplicated <- df_no_duplicated[, apply(df, 2, var) != 0]

# Data reduction
nearZeroVar(df_no_duplicated, name = TRUE)

# Collinearity
corr <- cor(df_no_duplicated[c(1:196)])
highCorr <- findCorrelation(corr, cutoff = 0.7, names = TRUE)
length(highCorr)
highCorr

#cfs
subset <- cfs(o_bullied ~., df_no_duplicated)
bone.cfs <- as.simple.formula(subset, "o_bullied")
bone.cfs

# information gain
df_no_duplicated <- as.data.frame(unclass(df_no_duplicated),
                                  stringsAsFactors = TRUE)
df_no_duplicated$o_bullied <- factor(df_no_duplicated$o_bullied)
bone.infogain <- InfoGainAttributeEval(o_bullied ~., data = df_no_duplicated)
sorted.features <- sort(bone.infogain, decreasing = TRUE)
sorted.features[1:10]

# Chi-square Test
ct <- table(df_no_duplicated$V2021, df_no_duplicated$o_bullied)
chisq.test(ct)

# Scaling the data
df1 <- scale(df)
df1
sum(is.na(df1))

df2 <- scale(df_no_duplicated)
sum(is.na(df2))


# Balanced Dataset -> undersampling

df_no_duplicated$o_bullied <- as.factor(df_no_duplicated$o_bullied)

balanced_df <- ovun.sample(o_bullied ~ ., data = df_no_duplicated, 
                                  method = "both", p=0.5, N=4946, seed = 1)$data

table(balanced_df$o_bullied)

############################################################
## Finish preprocessing --> preprocessing csv file

write.csv(df_no_duplicated, "preprocessed_data.csv")


############################################################
## Calculate measures --> From L5.R
calculate_measures <- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k)
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

############################################################
## Naive Bayes

# build a model
df_no_duplicated$o_bullied <- factor(df_no_duplicated$o_bullied) 
model_nb <- naiveBayes(o_bullied ~ ., data = df_no_duplicated)
model_nb

#Splitting Data into train and test
set.seed(31)
split <- initial_split(df_no_duplicated, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)
table(train$o_bullied)

## Balancing train data
train <- ovun.sample(o_bullied ~ ., data = train, 
                           method = "both", p=0.5, N=3263, seed = 123)$data
table(train$o_bullied)
table(test$o_bullied)
############################################################
## J48
modelLookup("J48")

# repeat 10 fold cross with 5 different models
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              summaryFunction = defaultSummary)
J48Grid <- expand.grid(C = c(0.01, 0.25, 0.5), M = (1:4))
J48.bullied <- train(o_bullied ~ ., data=train, method = "J48", 
                   trControl = train_control, tuneGrid = J48Grid)

# test 
pred <- predict(J48.bullied, newdata = test)
performance_measures48  <- confusionMatrix(pred, test$o_bullied)
performance_measures48

# ROC curve
predJ48_prob <- predict(J48.bullied, newdata = test, type = "prob")
roc_curveJ48 <- roc(response = test$o_bullied, predictor = predJ48_prob[,2])

aucJ48 <- auc(roc_curveJ48)
aucJ48
plot(roc_curveJ48, main = paste("AUC =", round(aucJ48, 2)))



cm0 <- performance_measures48$table
tp0 = cm0[1,1]
fp0 = cm0[1,2]
tn0 = cm0[2,2]
fn0 = cm0[2,1]

pm480 = calculate_measures(tp0, fp0, tn0, fn0)
pm480

cm1 <- performance_measures48$table
tp1 = cm1[2,2]
fp1 = cm1[2,1]
tn1 = cm1[1,1]
fn1 = cm1[1,2]

pm481 = calculate_measures(tp1, fp1, tn1, fn1)
pm481

# Calculate support for each class
support0 <- sum(cm0[1, ])
support1 <- sum(cm1[2, ])

# Calculate weighted averages
weighted_averages <- list()
for (measure in names(pm480)) {
  if (is.numeric(pm480[[measure]]) && is.numeric(pm481[[measure]])) {
    weighted_averages[[measure]] <- (pm480[[measure]] * support0 + pm481[[measure]] * support1) / (support0 + support1)
  }
}

# For ROC AUC, since it's a single value, you can assign it directly
weighted_averages[["ROC"]] <- aucJ48

# Print the weighted averages
print(weighted_averages)



############################################################
#  C5.0 # Perform this algorism
set.seed(31)
split <- initial_split(df_no_duplicated, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)
table(train$o_bullied)
## Balancing tarin data
train <- ovun.sample(o_bullied ~ ., data = train, 
                     method = "both", p=0.5, N=3263, seed = 123)$data
table(train$o_bullied)
table(test$o_bullied)

C5.bullied <- C5.0(o_bullied ~ ., data = train)

# test 
predC5 <- predict(C5.bullied, newdata = test, type = "class")
performance_measuresC5  <- confusionMatrix(data=predC5,
                                         reference = test$o_bullied)
performance_measuresC5

# ROC Curve
predC5_prob <- predict(C5.bullied, newdata = test, type = "prob")
roc_curveC5 <- roc(response = test$o_bullied, predictor = predC5_prob[,2])

aucC5 <- auc(roc_curveC5)
aucC5
plot(roc_curveC5, main = paste("AUC =", round(aucC5, 2)))

cm0 <- performance_measuresC5$table
tp0 = cm0[1,1]
fp0 = cm0[1,2]
tn0 = cm0[2,2]
fn0 = cm0[2,1]

pmC50 = calculate_measures(tp0, fp0, tn0, fn0)
pmC50

cm1 <- performance_measuresC5$table
tp1 = cm1[2,2]
fp1 = cm1[2,1]
tn1 = cm1[1,1]
fn1 = cm1[1,2]

pmC51 = calculate_measures(tp1, fp1, tn1, fn1)
pmC51

# Calculate support for each class
support0 <- sum(cm0[1, ])
support1 <- sum(cm1[2, ])

# Calculate weighted averages
weighted_averages <- list()
for (measure in names(pmC50)) {
  if (is.numeric(pmC50[[measure]]) && is.numeric(pmC51[[measure]])) {
    weighted_averages[[measure]] <- (pmC50[[measure]] * support0 + pmC51[[measure]] * support1) / (support0 + support1)
  }
}

# For ROC AUC, since it's a single value, you can assign it directly
weighted_averages[["ROC"]] <- aucC5

# Print the weighted averages
print(weighted_averages)
############################################################
##  rpart

# build 
rpart.bullied <- rpart(o_bullied ~ ., data = train, method = "class",
                       parms = list(split = "information"))
# plot the full tree (overfitted)
prp(rpart.bullied)

# test on training dataset
predRP <- predict(rpart.bullied, newdata = train, type = "class")
performance_measuresRP  <- confusionMatrix(data=predRP, 
                                         reference = train$o_bullied)
performance_measuresRP

# test on test dataset
predRP <- predict(rpart.bullied, newdata = test, type = "class")
performance_measuresRP  <- confusionMatrix(data=predRP, 
                                         reference = test$o_bullied)
performance_measuresRP

# ROC Curve
predRP_prob <- predict(rpart.bullied, newdata = test, type = "prob")
roc_curveRP <- roc(response = test$o_bullied, predictor = predRP_prob[,2])

aucRP <- auc(roc_curveRP)
aucRP
plot(roc_curveRP, main = paste("AUC =", round(aucRP, 2)))

performance_measuresRP$table

rpart.bullied$cptable

cm0 <- performance_measuresRP$table
tp0 = cm0[1,1]
fp0 = cm0[1,2]
tn0 = cm0[2,2]
fn0 = cm0[2,1]

pmRP0 = calculate_measures(tp0, fp0, tn0, fn0)
pmRP0

cm1 <- performance_measuresRP$table
tp1 = cm1[2,2]
fp1 = cm1[2,1]
tn1 = cm1[1,1]
fn1 = cm1[1,2]

pmRP1 = calculate_measures(tp1, fp1, tn1, fn1)
pmRP1
# Calculate support for each class
support0 <- sum(cm0[1, ])
support1 <- sum(cm1[2, ])

# Calculate weighted averages
weighted_averages <- list()
for (measure in names(pmRP0)) {
  if (is.numeric(pmRP0[[measure]]) && is.numeric(pmRP1[[measure]])) {
    weighted_averages[[measure]] <- (pmRP0[[measure]] * support0 + pmRP1[[measure]] * support1) / (support0 + support1)
  }
}

# For ROC AUC, since it's a single value, you can assign it directly
weighted_averages[["ROC"]] <- aucRP

# Print the weighted averages
print(weighted_averages)

############################################################
## Logistic
set.seed(31)
split <- initial_split(df_no_duplicated, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)
table(train$o_bullied)

## Balancing tarin data
train <- ovun.sample(o_bullied ~ ., data = train, 
                     method = "both", p=0.5, N=3263, seed = 123)$data
table(train$o_bullied)

control_parms <- glm.control(maxit = 100)

logitModel <- glm(train$o_bullied ~ ., data = train, 
                  family = "binomial", control = control_parms)

options(scipen=999)
summary(logitModel)

logitModel.pred <- predict(logitModel, test[, -197], type = "response")
data.frame(actual = test$o_bullied[1:10], 
           predicted = logitModel.pred[1:10])
pred <- factor(ifelse(logitModel.pred >= 0.5, 1, 0))
pred

performance_measures  <- confusionMatrix(pred, 
                                         test$o_bullied)
performance_measures

# ROC Curve
predLG_prob <- predict(logitModel, newdata = test, type = "response")
roc_curveLG <- roc(response = test$o_bullied, predictor = predLG_prob)

aucLG <- auc(roc_curveLG)
aucLG
plot(roc_curveLG, main = paste("AUC =", round(aucLG, 2)))

cm0 <- performance_measures$table
tp0 = cm0[1,1]
fp0 = cm0[1,2]
tn0 = cm0[2,2]
fn0 = cm0[2,1]

pmLR0 = calculate_measures(tp0, fp0, tn0, fn0)
pmLR0

cm1 <- performance_measures$table
tp1 = cm1[2,2]
fp1 = cm1[2,1]
tn1 = cm1[1,1]
fn1 = cm1[1,2]

pmLR1 = calculate_measures(tp1, fp1, tn1, fn1)
pmLR1
# Calculate support for each class
support0 <- sum(cm0[1, ])
support1 <- sum(cm1[2, ])

# Calculate weighted averages
weighted_averages <- list()
for (measure in names(pmLR0)) {
  if (is.numeric(pmLR0 [[measure]]) && is.numeric(pmLR1[[measure]])) {
    weighted_averages[[measure]] <- (pmLR0 [[measure]] * support0 + pmLR1[[measure]] * support1) / (support0 + support1)
  }
}

# For ROC AUC, since it's a single value, you can assign it directly
weighted_averages[["ROC"]] <- aucLG

# Print the weighted averages
print(weighted_averages)
## Logistic looks not good
############################################################


############################################################
# KNN
set.seed(31)
split <- initial_split(df_no_duplicated, prop = 0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)
table(train$o_bullied)

## Balancing tarin data
train <- ovun.sample(o_bullied ~ ., data = train, 
                     method = "both", p=0.5, N=3263, seed = 123)$data
table(train$o_bullied)
sapply(train, class)
sapply(test, class)

train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 2,
                              summaryFunction = defaultSummary)

knnModel <- train(o_bullied ~., data = train, method = "knn",
                  preProcess = c("center", "scale"), 
                  trControl = train_control)

predKNN <- predict(knnModel, newdata = test)

performance_measuresKNN <- confusionMatrix(predKNN, test$o_bullied)
performance_measuresKNN

# ROC Curve
predKNN_prob <- predict(knnModel, newdata = test, type = "prob")
roc_curveKNN <- roc(response = test$o_bullied, predictor = predKNN_prob[,2])

aucKNN <- auc(roc_curveKNN)
aucKNN
plot(roc_curveKNN, main = paste("AUC =", round(aucKNN, 2)))

cm0 <- performance_measuresKNN$table
tp0 = cm0[1,1]
fp0 = cm0[1,2]
tn0 = cm0[2,2]
fn0 = cm0[2,1]

pmKNN0 = calculate_measures(tp0, fp0, tn0, fn0)
pmKNN0

cm1 <- performance_measuresKNN$table
tp1 = cm1[2,2]
fp1 = cm1[2,1]
tn1 = cm1[1,1]
fn1 = cm1[1,2]

pmKNN1 = calculate_measures(tp1, fp1, tn1, fn1)
pmKNN1
# Calculate support for each class
support0 <- sum(cm0[1, ])
support1 <- sum(cm1[2, ])

# Calculate weighted averages
weighted_averages <- list()
for (measure in names(pmKNN0)) {
  if (is.numeric(pmKNN0 [[measure]]) && is.numeric(pmKNN1[[measure]])) {
    weighted_averages[[measure]] <- (pmKNN0 [[measure]] * support0 + pmKNN1[[measure]] * support1) / (support0 + support1)
  }
}

# For ROC AUC, since it's a single value, you can assign it directly
weighted_averages[["ROC"]] <- aucKNN

# Print the weighted averages
print(weighted_averages)

## KNN shows low TPR for class 0 $ class 1
###########################################################
#############################################################
## nnet
df <- read.csv("project_dataset.csv")
df[duplicated(df) | duplicated(df, fromLast = TRUE), ]
df_no_duplicated <- df[!duplicated(df),]
df_no_duplicated <- df_no_duplicated[, apply(df, 2, var) != 0]
nearZeroVar(df_no_duplicated, name = TRUE)
corr <- cor(df_no_duplicated[c(1:196)])
highCorr <- findCorrelation(corr, cutoff = 0.7, names = TRUE)
length(highCorr)
highCorr
subset <- cfs(o_bullied ~., df_no_duplicated)
bone.cfs <- as.simple.formula(subset, "o_bullied")
bone.cfs
df_no_duplicated <- as.data.frame(unclass(df_no_duplicated),
                                  stringsAsFactors = TRUE)
df_no_duplicated$o_bullied <- factor(df_no_duplicated$o_bullied)
bone.infogain <- InfoGainAttributeEval(o_bullied ~., data = df_no_duplicated)
sorted.features <- sort(bone.infogain, decreasing = TRUE)
sorted.features[1:10]

df_no_duplicated$o_bullied <- as.factor(df_no_duplicated$o_bullied)
df_no_duplicated$o_bullied <- ifelse(df_no_duplicated$o_bullied ==  1, "Y", "N")
sapply(df_no_duplicated, class)
df_no_duplicated$o_bullied <- as.factor(df_no_duplicated$o_bullied)

table(df_no_duplicated$o_bullied)

set.seed(31)
split <- initial_split(df_no_duplicated, prop=0.66, strata = o_bullied)
train <- training(split)
test <- testing(split)

## Balancing tarin data
train <- ovun.sample(o_bullied ~ ., data = train, 
                     method = "both", p=0.5, N=3263, seed = 123)$data
table(train$o_bullied)
sapply(train, class)
sapply(test, class)

ctr <- trainControl(method = "CV", number = 10, 
                    summaryFunction = twoClassSummary,
                    classProbs = TRUE, 
                    savePredictions = TRUE)

nnetGrid <- expand.grid(size = 1:10, decay = c(0, .1, 1, 2))
nnetFit <- train(x = train[, -10], y = train$o_bullied,
                 method = "nnet", metric = "ROC",
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 100,
                 MaxNWts = 1000,
                 trControl = ctr)
nnetFit

nnetFit$pred <- merge(nnetFit$pred, nnetFit$bestTune)
performance_measuresNNET <- confusionMatrix(nnetFit)
performance_measuresNNET

# ROC Curve
# Prediction probabilities for neural network model
predNNET_prob <- predict(nnetFit, newdata = test, type = "prob")

roc_curveNNET <- roc(response = test$o_bullied, predictor = predNNET_prob[, "Y"],
                     levels = c("N", "Y"), direction = "<")

aucNNET <- auc(roc_curveNNET)

plot(roc_curveNNET, main = paste("AUC =", round(aucNNET, 2)))


cm0 <- performance_measuresNNET$table
tp0 = cm0[1,1]
fp0 = cm0[1,2]
tn0 = cm0[2,2]
fn0 = cm0[2,1]

pmNNET0 = calculate_measures(tp0, fp0, tn0, fn0)
pmNNET0

cm1 <- performance_measuresNNET$table
tp1 = cm1[2,2]
fp1 = cm1[2,1]
tn1 = cm1[1,1]
fn1 = cm1[1,2]

pmNNET1 = calculate_measures(tp1, fp1, tn1, fn1)
pmNNET1

# Calculate support for each class
support0 <- sum(cm0[1, ])
support1 <- sum(cm1[2, ])

# Calculate weighted averages
weighted_averages <- list()
for (measure in names(pmNNET0)) {
  if (is.numeric(pmNNET0 [[measure]]) && is.numeric(pmNNET1[[measure]])) {
    weighted_averages[[measure]] <- (pmNNET0 [[measure]] * support0 + pmNNET1[[measure]] * support1) / (support0 + support1)
  }
}

# For ROC AUC, since it's a single value, you can assign it directly
weighted_averages[["ROC"]] <- aucNNET

# Print the weighted averages
print(weighted_averages)
## NNET Looks great
###########################################################################





