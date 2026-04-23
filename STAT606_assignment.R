#############################################################################
# STAT606 Assignment - Binary Classification Project
# Divorce Prediction Model
# Authors: Lungani Zungu (219024060) & Ntobeko Hlongwa()
#############################################################################

# ----------------------------- #
# 1. Load Libraries
# ----------------------------- #

library(dplyr)
library(caTools)
library(caret)
library(pROC)
library(h2o)
library(rpart)
library(rpart.plot)

options(scipen = 999)

# ----------------------------- #
# 2. Load Data
# ----------------------------- #

df <- read.csv("divorce_df.csv")

# ----------------------------- #
# 3. Data Understanding & Cleaning
# ----------------------------- #

str(df)
summary(df)

# Convert categorical variables if needed
df <- df %>%
  mutate(across(where(is.character), as.factor))

# ----------------------------- #
# 4. Define Target Variable
# ----------------------------- #

target <- names(df)[ncol(df)]   # assumes last column is target

# ----------------------------- #
# 5. Train-Test Split (70:30)
# ----------------------------- #

set.seed(606)

split <- sample.split(df[[target]], SplitRatio = 0.7)

train <- subset(df, split == TRUE)
test  <- subset(df, split == FALSE)

# ----------------------------- #
# 6. Initialize H2O
# ----------------------------- #

h2o.init()

train_h2o <- as.h2o(train)
test_h2o  <- as.h2o(test)

predictors <- setdiff(names(train), target)

# ----------------------------- #
# 7. Model 1: Logistic Regression
# ----------------------------- #

lr <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  family = "binomial"
)

# ----------------------------- #
# 8. Model 2: Decision Tree
# ----------------------------- #

dt <- rpart(
  as.formula(paste(target, "~ .")),
  data = train,
  method = "class"
)

# ----------------------------- #
# 9. Model 3: Naive Bayes (placeholder)
# ----------------------------- #

# add later if required

# ----------------------------- #
# 10. Predictions (Example: Logistic Regression)
# ----------------------------- #

pred_lr <- h2o.predict(lr, test_h2o)

# ----------------------------- #
# 11. Evaluation Template
# ----------------------------- #

# Example structure (reuse for all models)

# confusionMatrix(...)
# roc(...)
# auc(...)

# ----------------------------- #
# 12. Shutdown H2O
# ----------------------------- #

h2o.shutdown(prompt = FALSE)