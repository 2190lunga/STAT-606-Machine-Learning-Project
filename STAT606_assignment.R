#############################################################################
# STAT606 Assignment - Binary Classification Project
# Divorce Prediction Model
# Authors: Lungani Zungu (219024060) & Ntobeko Hlongwa()
#############################################################################

# ----------------------------------------------------------------------------- #
# 0. Load Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------------- #

library(dplyr)
library(caTools)
library(caret)
library(pROC)
library(h2o)
library(rpart)
library(rpart.plot)

options(scipen = 999)

# ----------------------------------------------------------------------------- #
# 1. Setup & User Parameters ---------------------------------------------------
# ----------------------------------------------------------------------------- #

seed <- 606
train_frac <- 0.7
metric <- "F1"
folds <- 5

# ----------------------------------------------------------------------------- #
# 2. Load, Inspect and Format Data --------------------------------------------
# ----------------------------------------------------------------------------- #

df <- read.csv("divorce_df.csv")

# Look at the structure of the data (variable names, types, first values):
str(df)

# Summary statistics for all variables:
summary(df)

# Number of rows and columns:
dim(df)

# Convert all character (text) variables to factors:
df <- df %>%
  mutate(across(where(is.character), as.factor))

# Convert binary 0/1 variables to factors (they are categories, not numbers):
df$cultural_background_match <- factor(df$cultural_background_match)
df$mental_health_issues <- factor(df$mental_health_issues)
df$infidelity_occurred <- factor(df$infidelity_occurred)
df$counseling_attended <- factor(df$counseling_attended)
df$pre_marital_cohabitation <- factor(df$pre_marital_cohabitation)
df$domestic_violence_history <- factor(df$domestic_violence_history)
df$divorced <- factor(df$divorced)

# Check the structure again after conversions:
summary(df)

# Check number of levels for each factor variable (looking for high cardinality):
sapply(Filter(is.factor, df), nlevels)

# Check each categorical variable for sparse categories:
table(df$education_level)
table(df$employment_status)
table(df$religious_compatibility)
table(df$conflict_resolution_style)
table(df$marriage_type)

# Check for missing values:
colSums(is.na(df))

# Check class balance of the target variable:
summary(df$divorced)
round(prop.table(table(df$divorced)) * 100, 2)

# ----------------------------------------------------------------------------- #
# 3. Specify df and target -----------------------------------------------------
# ----------------------------------------------------------------------------- #

target <- "divorced"

# ----------------------------------------------------------------------------- #
# 4. Train/Test Split ----------------------------------------------------------
# ----------------------------------------------------------------------------- #

set.seed(seed)

split <- sample.split(df[[target]], SplitRatio = train_frac)

training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# ----------------------------------------------------------------------------- #
# 5. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init()

train_h2o <- as.h2o(training_set)
test_h2o <- as.h2o(test_set)

# ----------------------------------------------------------------------------- #
# 6. Specify the attribute names -----------------------------------------------
# ----------------------------------------------------------------------------- #

predictors <- setdiff(names(training_set), target)

# ----------------------------------------------------------------------------- #
# 7. Model 1: Naive Bayes -----------------------------------------------------
# ----------------------------------------------------------------------------- #

########## --> Fit the model ----

nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 0,
  nfolds = folds,
  seed = seed
)

########## --> Extract predicted probabilities ----

preds_nb_train <- h2o.predict(nb, train_h2o)
preds_nb_test <- h2o.predict(nb, test_h2o)

preds_nb_train <- as.data.frame(preds_nb_train)
preds_nb_test <- as.data.frame(preds_nb_test)

train_nb_pred <- cbind(training_set,
                       setNames(preds_nb_train[, 3, drop = FALSE], "pred_prob"))

test_nb_pred <- cbind(test_set,
                      setNames(preds_nb_test[, 3, drop = FALSE], "pred_prob"))

########## --> Specify threshold ----

threshold <- 0.5

########## --> Determine the predicted class labels ----

train_nb_pred$pred_class <- factor(ifelse(train_nb_pred$pred_prob > threshold, "1", "0"))
test_nb_pred$pred_class <- factor(ifelse(test_nb_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# Training set
confusionMatrix(
  train_nb_pred$pred_class,
  train_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)

roc_nb_train <- roc(train_nb_pred[[target]], train_nb_pred$pred_prob)
auc(roc_nb_train)
plot(roc_nb_train)

# Test set
confusionMatrix(
  test_nb_pred$pred_class,
  test_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)

roc_nb_test <- roc(test_nb_pred[[target]], test_nb_pred$pred_prob)
auc(roc_nb_test)
plot(roc_nb_test)