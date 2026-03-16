### ----------------------------------------------------------------------- ###
# Feature Engineering for Predicting U.S. Presidential Election Results
# Author: Tess Wellington
# Course: STATS 101C @ UCLA
# Date: July 2024
# Description:
# This script performs feature engineering for predictive model
### ----------------------------------------------------------------------- ###

### ------------------ ### LIBRARIES / FILES ### ------------------ ###

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

### ------------- ### FEATURE ENGINEERING ### --------------- ###

train_subset <-
  train %>%
  dplyr::select(-name) # name is not in test set

## ---- recipes ---- ##

# main recipe
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
  step_corr(all_numeric_predictors(), threshold = tune()) %>%
  step_dummy(all_nominal_predictors()) %>% # needed for xgboost, nnet engines
  step_zv(all_numeric_predictors()) # needed for nnet engines

# interaction recipe 1
int_recipe <-
  recipe(winner ~ gdp_2020 + income_per_cap_2020, data = train) %>%
  step_log(gdp_2020, income_per_cap_2020) %>%
  step_interact(~gdp_2020:income_per_cap_2020)

# interaction recipe 2
inter_recipe <-
  recipe(winner ~ ., data = train_subset) %>%
  update_role(all_predictors(), 
              new_role = "other") %>% # update all to non-predictors
  update_role(id, 
              new_role = "ID") %>% # update id role
  update_role(total_votes, x0001e, x0086e, x2013_code, gdp_2020, income_per_cap_2020,
              new_role = "predictor") %>% # bring back the non proportion predictors
  step_log(gdp_2020, income_per_cap_2020) %>%
  step_num2factor(x2013_code, levels = c("1", "2", "3", "4", "5", "6"), ordered = TRUE) %>%
  step_mutate(male_18_prop = x0026e / x0025e) %>%
  step_mutate(over_45_prop = (x0012e + x0013e + x0014e + x0024e) / x0025e) %>%
  step_mutate(white_prop = x0064e / x0001e) %>%
  step_mutate(black_prop = x0065e / x0001e) %>%
  step_mutate(amerind_prop = x0066e / x0001e) %>%
  step_mutate(asian_prop = x0067e / x0001e) %>%
  step_mutate(nhpi_prop = x0068e / x0001e) %>%
  step_mutate(hispanic_prop = x0071e / x0001e) %>%
  step_mutate(citizen_18_prop = x0087e / x0025e) %>%
  step_mutate(no_hs_prop = (x0025e - c01_014e - c01_003e - c01_004e - c01_005e)/ x0025e)  %>% 
  step_mutate(bachelor_prop = c01_015e / x0025e) %>%
  step_impute_knn(gdp_2020, impute_with = imp_vars(all_predictors())) %>%
  step_interact(~gdp_2020:income_per_cap_2020) %>%   # gdp:income
  step_interact(~total_votes:x0001e) %>%   # votes:pop
  step_interact(~white_prop:bachelor_prop) %>% # white prop, bachelor
  step_corr(all_numeric_predictors(), threshold = tune()) %>%
  step_dummy(all_nominal_predictors()) %>% # needed for xgboost, nnet engines
  step_zv(all_numeric_predictors()) # needed for nnet engines

# upsample recipe
upsample_recipe <-
  recipe(winner ~ ., data = train_subset) %>%
  step_upsample(winner, over_ratio = 0.5) %>%
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
  step_corr(all_numeric_predictors(), threshold = tune()) %>%
  step_dummy(all_nominal_predictors()) %>% # needed for xgboost, nnet engines
  step_zv(all_numeric_predictors()) # needed for nnet engines

# downsample recipe
downsample_recipe <-
  recipe(winner ~ ., data = train_subset) %>%
  step_downsample(winner, under_ratio = 3) %>%
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
  step_corr(all_numeric_predictors(), threshold = tune()) %>%
  step_dummy(all_nominal_predictors()) %>% # needed for xgboost, nnet engines
  step_zv(all_numeric_predictors()) # needed for nnet engines

# pca recipe
pca_recipe <-
  main_recipe %>%
  step_pca(all_numeric_predictors(), threshold = tune("pca"))

# filter recipe
filter_recipe <-
  main_recipe %>%
  step_corr(all_predictors(), threshold = tune())

# mlp recipe
mlp_recipe <-
  main_recipe %>%
  step_normalize(all_numeric_predictors()) %>% # try with all
  step_zv(all_predictors())

# mlp w/ interaction recipe
mlp_inter_recipe <-
  inter_recipe %>%
  step_normalize(all_numeric_predictors()) %>%
  step_zv(all_predictors())

# glm recipe
logistic_mod <-
  logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# knn w/ PCA recipe:
pca_knn_recipe <-   
  main_recipe %>%
  na.omit() %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold = tune("pca"))

# glm w/ interaction fit:
logistic_fit <-
  logistic_workflow %>%
  fit(data = train)

logistic_fit %>% tidy()
