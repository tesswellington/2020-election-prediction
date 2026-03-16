### ----------------------------------------------------------------------- ###
# Exploratory Data Analysis for Predicting U.S. Presidential Election Results
# Author: Tess Wellington
# Course: STATS 101C @ UCLA
# Date: July 2024
# Description:
# This script performs exploratory data analysis and creates visualizations
# for demographic, economic, and voting data from U.S. counties to identify 
# patterns and relationships associated with the 2020 U.S. presidential election 
# results. These insights will guide feature selection and model development 
# for predicting county-level election outcomes.
### ----------------------------------------------------------------------- ###

### ------------------ ### LIBRARIES / FILES ### ------------------ ###

library(tidyverse)
library(tidymodels)
library(ggplot2)
library(patchwork)

train <- read_csv("train_class.csv")
test <- read_csv("test_class.csv")

### ------------- ### EXPLORATORY DATA ANALYSIS ### --------------- ###

train$winner_code <- ifelse(train$winner == "Biden", 1, 0)

train_nona <-
  train %>%
  na.omit()

# summary statistics

glimpse(train)
summary(train)

### ---- GDP and income:

plot(train_nona$winner_code, train_nona$gdp_2020)
cor.test(train_nona$winner_code, log(train_nona$gdp_2020))

gdp <- ggplot(train, aes(x = log(gdp_2020), y = winner_code, col = winner)) +
  geom_point() +
  labs(x = "Log of 2020 Gdp by County", y = "Winner Code") +
  ggtitle("Winner of County by Log of 2020 GDP") +
  scale_color_manual(values = c("dodgerblue2", "firebrick3"))

plot(train_nona$winner_code, train_nona$income_per_cap_2020)
cor.test(train_nona$winner_code, log(train_nona$income_per_cap_2020))

income <- ggplot(train, aes(x = log(income_per_cap_2020), y = winner_code, col = winner)) +
  geom_point() +
  labs(x = "Log of Income per Capita in 2020 by County", y = "Winner Code") +
  ggtitle("Winner of County by Log of Income per Capita in 2020") +
  scale_color_manual(values = c("dodgerblue2", "firebrick3"))

gdp + income

### ---- ages:

age_45_over_prop <- (train$x0012e + train$x0013e + train$x0014e + train$x0024e) / train$x0025e
age_55_over_prop <- (train$x0013e + train$x0014e + train$x0024e) / train$x0025e
age_62_over_prop <- (train$x0014e + train$x0024e) / train$x0025e


ggplot(train, aes(x = age_62_over_prop, y = winner_code, col = winner)) +
  geom_point() +
  labs(x = "Proportion of People over Age 62", y = "Winner Code") +
  ggtitle("Winner of County by Proportion of County over Age 62") +
  scale_color_manual(values = c("dodgerblue2", "firebrick3"))

 ggplot(train, aes(x = age_55_over_prop, y = winner_code, col = winner)) +
  geom_point() +
  labs(x = "Proportion of People over Age 55", y = "Winner Code") +
  ggtitle("Winner of County by Proportion of County over Age 55") +
  scale_color_manual(values = c("dodgerblue2", "firebrick3"))

ggplot(train, aes(x = age_62_over_prop, y = winner_code, col = winner)) +
  geom_point() +
  labs(x = "Proportion of People over Age 62", y = "Winner Code") +
  ggtitle("Winner of County by Proportion of County over Age 62") +
  scale_color_manual(values = c("dodgerblue2", "firebrick3"))

which(age_45_over_prop > 0.62 & train$winner == "Biden")
train$name[611]
train$total_votes[611]

train$name[975]
train$total_votes[975]

train$name[1671]
train$total_votes[1671]

sum(train$total_votes[611], train$total_votes[975], train$total_votes[1671])

sum(train$total_votes[which(age_45_over_prop > 0.60 & train$winner == "Biden")])

sum(train$total_votes[which(age_45_over_prop > 0.60 & train$winner == "Trump")])

(total_votes_Biden <- sum(train$total_votes[which(train$winner == "Biden")]))
(total_votes_Trump <- sum(train$total_votes[which(train$winner == "Trump")]))
(diff <- total_votes_Biden - total_votes_Trump)


which(age_45_over_prop < 0.2) 
# 1575: Chattahoochee County, Georgia
train$name[1575]
# 1619: Madison County, Idaho
train$name[1619]

### ---- race proportion:

black_pop_by_county <- train$x0038e
total_pop_by_county <- train$x0001e
black_prop_by_county <- black_pop_by_county / total_pop_by_county

plot(black_prop_by_county, train$winner_code)

### ---- housing, population, votes:

votes <- ggplot(train, aes(x = total_votes, y = winner_code, col = winner)) +
  geom_point() +
  labs(x = "Total Votes", y = "Winner Code") +
  ggtitle("Winner of County vs Total Votes of County") +
  scale_color_manual(values = c("dodgerblue2", "firebrick3"))
cor.test(train_nona$winner_code, log(train_nona$total_votes))


population <- ggplot(train, aes(x = x0001e, y = winner_code, col = winner)) +
  geom_point() +
  labs(x = "Total Population", y = "Winner Code") +
  ggtitle("Winner of County vs Total Population of County") +
  scale_color_manual(values = c("dodgerblue2", "firebrick3"))
cor.test(train_nona$winner_code, log(train_nona$x0001e))


houses <- ggplot(train, aes(x = x0086e, y = winner_code, col = winner)) +
  geom_point() +
  labs(x = "Total Housing Units", y = "Winner Code") +
  ggtitle("Winner of County vs Total Housing Units of County") +
  scale_color_manual(values = c("dodgerblue2", "firebrick3"))
cor.test(train_nona$winner_code, log(train_nona$x0086e))


houses + votes + population

###

### ----------------------- ### FEATURES ### ---------------------- ###

## GENDER: 2-3, 26-27 ---------------------------------
# male 18+ proportion:                  x0026e / x0025e

## AGE: 5-31 ------------------------------------------
# age over 45 proportion:     (x0012e + x0013e + x0014e + x0024e) / x0025e

## RACE: 34-69 ----------------------------------------
# black or african american proportion: x0065e / x0001e
# american indian proportion:           x0066e / x0001e
# asian proportion:                     x0067e / x0001e
# native hawaiian and PI proportion:    x0068e / x0001e
# other race proportion:                x0069e / x0001e

## ETHNICITY: 71-85 -----------------------------------
# hispanic or latino proportion:        x0071e / x0001e

## CITIZEN: 87-89  ------------------------------------
# citizen, over 18 proportion:          x0087e / x0025e

## EDUCATION: C ---------------------------------------
# did not graduate HS proportion:    (x0025e - C01_014E - C01_003E - C01_004E - C01_005E)  / x0025e   
# bachelor's or more proportion:     (C01_005E + C01_015E) / x0025e

