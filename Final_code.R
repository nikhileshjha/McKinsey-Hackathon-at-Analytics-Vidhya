rm(list=ls())
library(data.table)
library(xgboost)
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(xgboost))
library(data.table)
library(woeBinning)

cat("Loading data...\n")
setwd("/Users/apple/Desktop/Analytics Vidhya Hackathon")
train <- read.csv("train_ajEneEa.csv")
test <- read.csv("test_v2akXPA.csv")
test_id <- test$id
set.seed(0)
train <- train[sample(1:nrow(train),nrow(train)), ]

#---------------------------
cat("Basic preprocessing...\n")
y <- train$stroke
tri <- 1:round(0.9*nrow(train))

# Using one hot encoding on categorical variables
gendert <- model.matrix(~train$gender-1,train$gender)
colnames(gendert) <- c("female","male","other")


train$hypertension <- as.factor(train$hypertension)

hypertensiont <- model.matrix(~train$hypertension,train$hypertension)
colnames(hypertensiont) <- c("nohypertension","hypertension")

train$heart_disease <- as.factor(train$heart_disease)
heart_diseaset <- model.matrix(~train$heart_disease,train$heart_disease)
colnames(heart_diseaset) <- c("no_heart_disease","heart_disease")

ever_married_t <- model.matrix(~train$ever_married-1,train$ever_married)
colnames(ever_married_t) <- c("never_married","ever_married")

work_type_t <- model.matrix(~train$work_type-1,train$work_type)
colnames(work_type_t) <- c("govt_job","never_worked","private","self-employed","child")

residence_type_t <- model.matrix(~train$Residence_type-1,train$Residence_type)
colnames(residence_type_t) <- c("rural","urban")

smoking_status_t <- model.matrix(~train$smoking_status-1,train$smoking_status)
colnames(smoking_status_t) <- c("unknown","formerly_smoked","never_smoked",
                                "smokes")





# Removing the target variable from dataset
train <- train %>%
  select(-id) %>%
  select(-gender) %>%
  select(-hypertension) %>%
  select(-heart_disease) %>%
  select(-ever_married) %>%
  select(-work_type) %>%
  select(-Residence_type) %>%
  select(-smoking_status) %>% 
  select(-stroke)

train <- cbind(train, gendert, hypertensiont,heart_diseaset,ever_married_t,work_type_t,residence_type_t,smoking_status_t)

#Performing all the transformation on test data
rm(gendert, hypertensiont,heart_diseaset,ever_married_t,work_type_t,residence_type_t,smoking_status_t)
# Using one hot encoding on categorical variables
gendert <- model.matrix(~test$gender-1,test$gender)
colnames(gendert) <- c("female","male","other")


test$hypertension <- as.factor(test$hypertension)

hypertensiont <- model.matrix(~test$hypertension,test$hypertension)
colnames(hypertensiont) <- c("nohypertension","hypertension")

test$heart_disease <- as.factor(test$heart_disease)
heart_diseaset <- model.matrix(~test$heart_disease,test$heart_disease)
colnames(heart_diseaset) <- c("no_heart_disease","heart_disease")

ever_married_t <- model.matrix(~test$ever_married-1,test$ever_married)
colnames(ever_married_t) <- c("never_married","ever_married")

work_type_t <- model.matrix(~test$work_type-1,test$work_type)
colnames(work_type_t) <- c("govt_job","never_worked","private","self-employed","child")

residence_type_t <- model.matrix(~test$Residence_type-1,test$Residence_type)
colnames(residence_type_t) <- c("rural","urban")

smoking_status_t <- model.matrix(~test$smoking_status-1,test$smoking_status)
colnames(smoking_status_t) <- c("unknown","formerly_smoked","never_smoked",
                                "smokes")

# Removing the target variable from dataset
test <- test %>%
  select(-id) %>%
  select(-gender) %>%
  select(-hypertension) %>%
  select(-heart_disease) %>%
  select(-ever_married) %>%
  select(-work_type) %>%
  select(-Residence_type) %>%
  select(-smoking_status)


test <- cbind(test, gendert, hypertensiont,heart_diseaset,ever_married_t,work_type_t,residence_type_t,smoking_status_t)

rm(gendert, hypertensiont,heart_diseaset,ever_married_t,work_type_t,residence_type_t,smoking_status_t)

#===================#
cat("Preparing data...\n")
dtest <- xgb.DMatrix(data = data.matrix(test))
tri <- caret::createDataPartition(y, p = 0.9, list = F)
dtrain <- xgb.DMatrix(data = data.matrix(train[tri,]), label = y[tri])
dval <- xgb.DMatrix(data = data.matrix(train[-tri,]), label = y[-tri])
cols <- colnames(dtrain)


cat("Training model...\n")
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          eta = 0.012,
          max_depth = 7,
          min_child_weight = 148,
          colsample_bytree = 1,
          gamma = 167.125,
          subsample = 0.6928,
          alpha = 43.2165,
          lambda = 74.1,
          scale_pos_weight = 98.19585,
          nrounds = 2000)

modelfinal <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 200)

(imp <- xgb.importance(cols, model=model36))
xgb.plot.importance(imp, top_n = 10)


cat("Creating submission file...\n")
pred <- predict(modelfinal, dtest)
pred <- as.numeric(pred > 0.5)

pred <- as.data.frame(pred)
sub  <- data.table(id = test_id, stroke = NA)
sub$stroke = pred
write.csv(sub, "sub_final.csv",row.names = FALSE)
rm(sub)

