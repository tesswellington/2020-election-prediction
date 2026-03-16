### ----------------------------------------------------------------------- ###
# Model Evaluation and Comparison for Predicting U.S. Presidential Election Results
# Author: Tess Wellington
# Course: STATS 101C @ UCLA
# Date: July 2024
#
### ----------------------------------------------------------------------- ###

### ------------------ ### LIBRARIES / FILES ### ------------------ ###

library(tidyverse)
library(tidymodels)
library(vip)  

train <- read_csv("train_class.csv")
test <- read_csv("test_class.csv")

### ------------------ ### MODEL EVALUATION ### ------------------ ###

boost_predictions <- read_csv("boosted_predictions.csv")

evaluation_data <- test %>%
  inner_join(boost_predictions, by = "id")

### --- confusion matrix ---

conf_matrix <- conf_mat(evaluation_data, truth = winner, estimate = .pred_class)

### --- classification metrics ---

classification_metrics <- metric_set(accuracy, precision, recall, f_meas, mcc)
classification_results <- classification_metrics(evaluation_data, truth = winner, estimate = .pred_class)

classification_results

### --- feature importance ---

vip(final_boosted_fit)

### --- ROC curve ---

roc_curve_data <- roc_curve(evaluation_data, truth = winner, .pred_Biden)
autoplot(roc_curve_data)

### --- AUC score ---

auc_score <- roc_auc(evaluation_data, truth = winner, .pred_Biden)
auc_score
