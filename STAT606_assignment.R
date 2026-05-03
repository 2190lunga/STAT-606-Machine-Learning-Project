#############################################################################
# STAT606 Assignment - Binary Classification Project
# Divorce Prediction Model
# Authors: Lungani Zungu (219024060) & Ntobeko Hlongwa(219021668)
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

# ----------------------------------------------------------------------------- #
# 8. Fit a decision tree using h2o --------------------------------------------
# ----------------------------------------------------------------------------- #


########## --> Hyperparameter tuning ----

hyper_params <- list(
  max_depth = seq(3, 21, by = 2), 
  min_rows = c(1, 5, 10, 20, 50)
  #min_split_improvement= 15
)

# Define search criteria:
search_criteria <- list(
  strategy = "Cartesian"
)


h2o.rm("dtree_grid")

grid <- h2o.grid(
  algorithm = "gbm",  # gradient boosting method with ntrees = 1)
  grid_id = "dtree_grid", # grid search ID
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  ntrees = 1,
  learn_rate = 1.0, # Full weight per tree 
  sample_rate = 1.0,  # use 100% of the training data
  col_sample_rate = 1.0, # use 100% of the attributes
  stopping_rounds = 0,
  seed = seed,
  nfolds =folds
)


model_results_dt <- h2o.getGrid("dtree_grid", sort_by = metric, decreasing = TRUE) 

print(model_results_dt)


best_model_id <- model_results_dt@model_ids[[1]] # Extract the best model ID 


best_model <- h2o.getModel(best_model_id)


tuned_param_names <- names(hyper_params)

best_tuned_values <- lapply(tuned_param_names, function(param_name) { 
  best_model@allparameters[[param_name]]
})

names(best_tuned_values) <- tuned_param_names

########## --> Fit the model ----

final_dt_model <- do.call(h2o.decision_tree, c(
  list(
    x = predictors,
    y = target,
    training_frame = train_h2o,
    seed = seed                      
  ),
  best_tuned_values     
))

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_dt_train <- h2o.predict(final_dt_model, train_h2o)
preds_dt_test <- h2o.predict(final_dt_model, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_dt_train <- as.data.frame(preds_dt_train)
preds_dt_test <- as.data.frame(preds_dt_test)

View(preds_dt_train)

train_dt_pred <- cbind(training_set,
                       setNames(preds_dt_train[, 3, drop = FALSE], "pred_prob")) 

test_dt_pred <- cbind(test_set, 
                      setNames(preds_dt_test[, 3, drop = FALSE], "pred_prob"))  


View(train_dt_pred)


threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_dt_pred$pred_class <- factor(ifelse(train_dt_pred$pred_prob > threshold,"1","0"))

# test
test_dt_pred$pred_class <- factor(ifelse(test_dt_pred$pred_prob > threshold, "1", "0"))

########## --> confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
confusionMatrix(
  train_dt_pred$pred_class,
  train_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_dt_train <- pROC::roc(train_dt_pred[[target]], train_dt_pred$pred_prob)
pROC::auc(roc_dt_train)
plot(roc_dt_train)


# test set

# predicted classes first then actual classes for the test set
confusionMatrix(
  test_dt_pred$pred_class,
  test_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_dt_test <- pROC::roc(test_dt_pred[[target]], test_dt_pred$pred_prob)
pROC::auc(roc_dt_test)
plot(roc_dt_test)

# ----------------------------------------------------------------------------- #
# 9. Fit a decision tree using rpart --------------------------------------------
# ----------------------------------------------------------------------------- #

set.seed(seed)


DT_rpart <- rpart(
  as.formula(paste(target, "~ .")),
  data = training_set,
  method = "class", # for classification
  xval = folds, # CV 
  control = rpart.control(
    #cp = 0.02,             # complexity parameter for pruning
    #minsplit = 20,         # minimum observations to attempt a split
    maxdepth = 4           # maximum depth
  )
) 

printcp(DT_rpart)
plotcp(DT_rpart)


### Extract predicted probabilities:


pred_prob_DT_train <- predict(DT_rpart, newdata = training_set, type = "prob")

train_DT_rpart <- cbind(training_set, 
                        setNames(data.frame(pred_prob_DT_train[, 2]), "pred_prob")) # probs in column 2 (for "1")

# View the results:

View(train_DT_rpart)


pred_prob_DT_test <- predict(DT_rpart, newdata = test_set, type = "prob")

test_DT_rpart <- cbind(test_set, 
                       setNames(data.frame(pred_prob_DT_test[, 2]), "pred_prob")) # We only want the probs in column 2 (for "1")


threshold <- 0.5

########## --> Determine the predicted class labels ----

# training set
train_DT_rpart$pred_class <- factor(ifelse(train_DT_rpart$pred_prob > threshold, "1","0"))

# test
test_DT_rpart$pred_class <- factor(ifelse(test_DT_rpart$pred_prob > threshold,"1","0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
confusionMatrix(
  train_DT_rpart$pred_class,
  train_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_DT_train_rpart <- pROC::roc(train_DT_rpart[[target]], train_DT_rpart$pred_prob)
pROC::auc(roc_DT_train_rpart)
plot(roc_DT_train_rpart)

# test set

# predicted classes first then actual classes for the test set
confusionMatrix(
  test_DT_rpart$pred_class,
  test_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_DT_test_rpart <- pROC::roc(test_DT_rpart[[target]], test_DT_rpart$pred_prob)
pROC::auc(roc_DT_test_rpart)
plot(roc_DT_test_rpart)


########## --> visualize the DT ----

dev.new(width = 15, height = 20) 

rpart.plot(DT_rpart)
rpart.plot(DT_rpart, yesno = 1, type = 2, fallen.leaves = ) 


# ----------------------------------------------------------------------------- #
# 10. Fit a logistic regression model --------------------------------------------
# ----------------------------------------------------------------------------- #
########## --> Fit the LR model ----

LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  family = "binomial",
  lambda = 0,
  compute_p_values = TRUE
)

########## --> LR inference: p-values & odds ratios ----

# extract the coefficients table (has columns: names, coefficients, std_error, z_value, p_value)
LR_results <- as.data.frame(h2o.coef_with_p_values(LR))

# odds ratio = e^coefficient — tells you how much the odds of divorce
# multiply for a 1-unit increase in that feature
LR_results$OR <- exp(LR_results$coefficients)

# round p-values to 4 decimal places for readability
LR_results$p_value <- round(LR_results$p_value, 4)

View(LR_results)

########## --> LR predictions ----

# predict on train and test — exactly the same pattern as NB and DT
preds_LR_train <- h2o.predict(LR, train_h2o)
preds_LR_test  <- h2o.predict(LR, test_h2o)

# convert H2O frames to regular R data.frames
preds_LR_train <- as.data.frame(preds_LR_train)
preds_LR_test  <- as.data.frame(preds_LR_test)

# attach the predicted probability of class "1" to the original split data
train_LR <- cbind(training_set,
                  setNames(preds_LR_train[, 3, drop = FALSE], "pred_prob"))
test_LR  <- cbind(test_set,
                  setNames(preds_LR_test[, 3, drop = FALSE], "pred_prob"))

########## --> LR threshold & class labels ----

threshold <- 0.5

train_LR$pred_class <- factor(ifelse(train_LR$pred_prob > threshold, "1", "0"))
test_LR$pred_class  <- factor(ifelse(test_LR$pred_prob  > threshold, "1", "0"))

########## --> LR confusion matrix ----

confusionMatrix(train_LR$pred_class, train_LR[[target]],
                positive = "1",
                mode = "everything"
)

confusionMatrix(test_LR$pred_class, test_LR[[target]],
                positive = "1",
                mode = "everything"
)

########## --> LR ROC & AUC ----

# TRAIN ROC
roc_LR_train <- pROC::roc(train_LR[[target]], train_LR$pred_prob)
pROC::auc(roc_LR_train)
plot(roc_LR_train)

# TEST ROC
roc_LR_test <- pROC::roc(test_LR[[target]], test_LR$pred_prob)
pROC::auc(roc_LR_test)
plot(roc_LR_test)


# ----------------------------------------------------------------------------- #
# 11. Combined ROC Curve (Test Set) --------------------------------------------
# ----------------------------------------------------------------------------- #

# plot the first ROC curve, then layer the others on top with lines()
plot(roc_nb_test, col = "blue", lwd = 2,
     main = "ROC Curves - Test Set Comparison")
lines(roc_dt_test, col = "red", lwd = 2)
lines(roc_DT_test_rpart, col = "green", lwd = 2)
lines(roc_LR_test, col = "purple", lwd = 2)

# add a legend so we know which colour is which model
legend("bottomright",
       legend = c(
         paste0("Naive Bayes (AUC = ", round(auc(roc_nb_test), 4), ")"),
         paste0("DT H2O (AUC = ", round(auc(roc_dt_test), 4), ")"),
         paste0("DT rpart (AUC = ", round(auc(roc_DT_test_rpart), 4), ")"),
         paste0("Logistic Reg (AUC = ", round(auc(roc_LR_test), 4), ")")
       ),
       col = c("blue", "red", "green", "purple"),
       lwd = 2
)


# ----------------------------------------------------------------------------- #
# 12. Shutdown H2O -------------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.shutdown(prompt = FALSE)