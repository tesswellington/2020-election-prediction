### ----------------------------------------------------------------------- ###
# Stacked Machine Learning Model
# Author: Tess Wellington
# Course: STATS 101C @ UCLA
# Date: July 2024
# Description:
# Machine Learning Model by Tess Wellington which stacks models made by multiple classmates.
# This model tied for 5th place out of 17 teams in accuracy for a course competition.
#
### ----------------------------------------------------------------------- ###

### ------------------ ### LIBRARIES / FILES ### ------------------ ###

library(tidyverse)
library(tidymodels)
library(ggplot2)
library(patchwork)
library(glmnet)
library(stacks)
library(nnet)     # for neural networks (mlp)
library(baguette) # for bagging neural networks (bag_mlp)
library(earth)    # for mars model
library(kknn)     # for impute_knn
library(xgboost)  # for boosted tree model
library(kernlab)  # for svm model
library(themis)   # for upsampling

library(yardstick)
library(probably)

train <- read_csv("train_class.csv")
test <- read_csv("test_class.csv")


# recipes + models + workflows of all stack candidates below (RUN THIS FIRST):

### ------------------------ ### BOOSTED v3 ### ------------------------- ###

train_subset <- train %>% select(-name)

boost3_recipe <-
  recipe(winner ~ ., data = train_subset) %>%
  update_role(all_predictors(), 
              new_role = "other") %>% # update all to non-predictors
  update_role(id, 
              new_role = "ID") %>% # update id role
  update_role(total_votes, x0001e, x0086e, x2013_code, gdp_2020,
              new_role = "predictor") %>% # bring back the non proportion predictors
  step_num2factor(x2013_code, levels = c("1", "2", "3", "4", "5", "6"), ordered = TRUE) %>%
  step_mutate(male_18_prop = x0026e / x0025e) %>%
  step_mutate(over_45_prop = (x0012e + x0013e + x0014e + x0024e) / x0025e) %>%
  step_mutate(black_prop = x0065e / x0001e) %>%
  step_mutate(ai_prop = x0066e / x0001e) %>%
  step_mutate(asian_prop = x0067e / x0001e) %>%
  step_mutate(nhpi_prop = x0068e / x0001e) %>%
  step_mutate(hispanic_prop = x0071e / x0001e) %>%
  step_mutate(citizen_18_prop = x0087e / x0025e) %>%
  step_mutate(no_hs_prop = (x0025e - c01_014e - c01_003e - c01_004e - c01_005e)/ x0025e)  %>% 
  step_mutate(bachelor_prop = c01_015e / x0025e) %>%
  step_impute_knn(gdp_2020, impute_with = imp_vars(all_predictors())) %>%
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

boost3_model <-
  boost_tree() %>% 
  set_engine("xgboost") %>%
  set_mode("classification")

boost3_workflow <- 
  workflow() %>%
  add_recipe(boost3_recipe) %>%
  add_model(boost3_model)

### --------------------- ### LOU SVM ### --------------------- ###

main_recipe <-
  recipe(winner ~ ., data = train_subset) %>%
  update_role(all_predictors(), 
              new_role = "other") %>% # update all to non-predictors
  update_role(id, 
              new_role = "ID") %>% # update id role
  update_role(total_votes, x0001e, x0086e, x2013_code, gdp_2020,
              new_role = "predictor") %>% # bring back the non proportion predictors
  step_num2factor(x2013_code, levels = c("1", "2", "3", "4", "5", "6"), ordered = TRUE) %>%
  step_mutate(male_18_prop = x0026e / x0025e) %>%
  step_mutate(over_45_prop = (x0012e + x0013e + x0014e + x0024e) / x0025e) %>%
  step_mutate(white_prop = x0064e / x0001e) %>%
  step_mutate(black_prop = x0065e / x0001e) %>%
  step_mutate(ai_prop = x0066e / x0001e) %>%
  step_mutate(asian_prop = x0067e / x0001e) %>%
  step_mutate(nhpi_prop = x0068e / x0001e) %>%
  step_mutate(hispanic_prop = x0071e / x0001e) %>%
  step_mutate(citizen_18_prop = x0087e / x0025e) %>%
  step_mutate(no_hs_prop = (x0025e - c01_014e - c01_003e - c01_004e - c01_005e)/ x0025e)  %>% 
  step_mutate(bachelor_prop = c01_015e / x0025e) %>%
  step_impute_knn(gdp_2020, impute_with = imp_vars(all_predictors())) %>%
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% # needed for xgboost, nnet engines
  step_zv(all_numeric_predictors()) # needed for nnet engines

lou_svm_model <-
  svm_rbf(cost = 32768,
          rbf_sigma = 0.001) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

lou_svm_workflow <-
  workflow() %>%
  add_recipe(main_recipe)%>%
  add_model(lou_svm_model)

### ------------------------ ### CLASS STACK v4 ### ------------------------- ###


# initializing seed & folds:

set.seed(22)
train_folds <- vfold_cv(train, v = 10, strat = winner)

ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

# candidates:

boost3_res  <- 
  fit_resamples(
    boost3_workflow,
    resamples = train_folds,
    metrics = metric_set(accuracy, roc_auc),
    control = ctrl_res
  )

lou_svm_res  <- 
  fit_resamples(
    lou_svm_workflow,
    resamples = train_folds,
    metrics = metric_set(accuracy, roc_auc),
    control = ctrl_res
  )

# creating the stack:

stacks()

class_stack <-
  stacks() %>%
  add_candidates(boost3_res) %>%
  add_candidates(lou_svm_res)

# blend predictions:

class_stack <-
  class_stack %>%
  blend_predictions() # gives stacking coeff to each model candidate (nonzero coeff become members)

# check stacking coefficient of each candidate:

collect_parameters(class_stack, "boost3_res")
collect_parameters(class_stack, "lou_svm_res")


# now we fit the members (candidates with nonzero stacking coeff):

class_stack <-
  class_stack %>%
  fit_members()


# predicting test data:

stack_preds <-
  test %>% 
  select(id) %>%
  bind_cols(predict(class_stack, new_data = test)) %>% 
  rename(winner = .pred_class)

write_csv(stack_preds, "class_stack_v5.csv")

