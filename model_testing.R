### ----------------------------------------------------------------------- ###
# Model Testing for Predicting U.S. Presidential Election Results
# Author: Tess Wellington
# Course: STATS 101C @ UCLA
# Date: July 2024
# Description:
# This script makes predictions on test data using fully trained models from
# 'model_training.R'
#
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

### --------------- ### TESTING FINAL MODEL ### --------------- ###

final_boosted_fit <- readRDS("final_boosted_model.rds")

# make predictions on test data
boost_predictions <- test %>% 
  dplyr::select(id) %>%
  bind_cols(predict(final_boosted_fit, new_data = test))

# formatting for submission
boost_predictions <-
  test %>% 
  dplyr::select(id) %>%
  bind_cols(predict(boost_fit, new_data = test))

# save predictions
write_csv(boost_predictions, "boosted_predictions.csv")

