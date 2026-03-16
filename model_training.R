### ----------------------------------------------------------------------- ###
# Model Training and Tuning for Predicting U.S. Presidential Election Results
# Author: Tess Wellington
# Course: STATS 101C @ UCLA
# Date: July 2024
# Description:
# This script includes model training, hyperparameter tuning, model fitting, and
# data predictions. More specifically, this script: 
# (1) defines models using built-in engines such as xgboost, nnet, and earth; 
# (2) implements workflows for recipes (previously defined in 
#     'feature_engineering.R') and models (defined below)
# (3) tunes hyperparameters using 10-fold cross-validation and tuning grids;
# (4) fits models to cross-validation resamples and collects performance
#     metrics
# (5) fits best model to full training data and saves final model
#
### ----------------------------------------------------------------------- ###

### ---------------------- ### LIBRARIES / FILES ### ---------------------- ###

library(tidyverse)
library(tidymodels)
library(ggplot2)
library(patchwork)
library(glmnet)
library(discrim)  # for discriminant analysis (LDA)
library(nnet)     # for neural networks (mlp)
library(baguette) # for bagging neural networks (bag_mlp)
library(earth)    # for mars model
library(ROSE)     # upsampling
library(themis)   # downsampling

train <- read_csv("train_class.csv")
test <- read_csv("test_class.csv")

### ---------------------- ### MODEL DEFINITIONS ### ------------------------ ###

# base boosted tree model using xgboost
boost_model <-
  boost_tree() %>% 
  set_engine("xgboost") %>%
  set_mode("classification")

# tunable boosted tree model
boost_model <-
  boost_tree(trees = tune(), min_n = tune()) %>% 
  set_engine("xgboost") %>%
  set_mode("classification")

# base mlp model
mlp_model <-
  mlp() %>%
  set_engine("nnet") %>%
  set_mode("classification")

# bagged mlp model:
mlpbag_model <-  
  bag_mlp(hidden_units = 5) %>%
  set_engine("nnet") %>%
  set_mode("classification")

# bagged mars model:
marsbag_model <-
  bag_mars() %>%
  set_engine("earth") %>%
  set_mode("classification")

# knn model w/ PCA:
pca_knn_model <- 
  nearest_neighbor(neighbors = tune(), weight_func = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")


### ------------------ ### WORKFLOWS / WORKFLOW SETS ### ------------------ ###

# glm w/ interaction workflow:
logistic_workflow <- 
  workflow() %>%
  add_recipe(int_recipe) %>%
  add_model(logistic_mod)

# boosted tree workflow:
boost_workflow <- 
  workflow() %>%
  add_recipe(main_recipe) %>%
  add_model(boost_model)

# boosted tree w/ interaction workflow:
boost_workflow <- 
  workflow() %>%
  add_recipe(inter_recipe) %>%
  add_model(boost_model)

# bagged mlp workflow:
mlpbag_workflow <-
  workflow() %>%
  add_recipe(main_recipe) %>%
  add_model(mlpbag_model)

# workflow set:
models_set <- 
  workflow_set(
    preproc = list(pca_knn = pca_knn_recipe),
    models = list(knn = pca_knn_model),
    cross = TRUE
  ) %>% anti_join(tibble(wflow_id = c("mlp_boost", "mlp_inter_boost", "mlp_marsbag", "mlp_inter_marsbag",
                                      "main_mlpbag", "inter_mlpbag"))) %>%
  option_add(control = control_grid(extract = function(x) x))

### ----------------- ### CROSS-VALIDATION RESAMPLES ### ------------------ ###

set.seed(22)
class_folds <- vfold_cv(train, v=10, strat = winner)
keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

### -------------------- ### HYPERPARAMETER TUNING ### --------------------- ###

models <- 
  models_set %>%
  workflow_map("tune_grid", resamples = class_folds, grid = 15, 
               metrics = metric_set(accuracy, roc_auc), verbose = TRUE)

autoplot(models, select_best = TRUE)

rank_results(boost_models, rank_metric = "accuracy", select_best=TRUE) %>%
  select(rank, mean, model, wflow_id, .config)

# finding best values:

boost_results <-
  models %>%
  extract_workflow_set_result("pca_knn_knn")

best_boost <- select_best(boost_results, metric="accuracy")

final_boosted_workflow <- finalize_workflow(boost_workflow, best_boost)

boost_wflows <-
  boost_results %>%
  select(id, .extracts) %>%
  unnest(cols = .extracts) %>%
  inner_join(best_boost)

boost_wflows # gives column(s) of best tunes (aka ideal hyperparameters)

# tuned:
# main_boost: accuracy 0.924  | thresh = 0.924 | trees = 33, min_n = 16
# upsample_boost: acc 0.912   | thresh = 0.924 | trees = 33, min_n = 16
# mlp_mlpbag: accuracy 0.919  | thresh = 0.929 | hidden = 5
# mlp_inter_mlpmap: acc 0.916 | thresh = 0.929 | hidden = 5


### ------------------ ### FITTING TO RESAMPLES ### ------------------ ###

# boosted tree:
boost_fit <-
  final_boosted_workflow %>%
  fit_resamples(resamples = class_folds, control = keep_pred)

boost_metrics <- collect_metrics(boost_fit)
autoplot(boost_fit)

# bagged mlp:
mlpbag_fit <-
  mlpbag_workflow %>%
  fit_resamples(resamples = class_folds, control = keep_pred)

mlpbag_metrics <- collect_metrics(mlpbag_fit)
autoplot(mlpbag_fit)


### --------------- ### FITTING TO FULL TRAINING DATA ### --------------- ###

# boosted tree:
final_boost_fit <-
  final_boosted_workflow %>%
  fit(data = train)

saveRDS(final_boost_fit, "final_boost_model.rds")

# bagged mlp:
mlpbag_fit <-
  mlpbag_workflow %>%
  fit(data = train)

saveRDS(mlpbag_fit, "mlpbag_model.rds")

