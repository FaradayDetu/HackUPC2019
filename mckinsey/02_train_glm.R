

# Load libraries and data

library(tidyverse)
library(glmnet)

train <- read.csv('Severity-Predictor/transformed-2.csv')



# use only numeric features
features <- train %>% select_if(is.numeric) %>% select(-target) %>% as.matrix()
target <- train %>% pull(target)


# Train lasso
lasso <- cv.glmnet(features, target, nfolds = 5, 
                   type.measure = 'auc', family = 'binomial')

# performance is pretty good!
plot(lasso$glmnet.fit)

# Use optimal lambda to retrain
simple_lasso <- glmnet(features, target, family = 'binomial', lambda = exp(-4))

df_coefs <- simple_lasso$beta %>%
  as.matrix %>% 
  as.data.frame()

names <- rownames(df_coefs)
values <- df_coefs[, 1]

names <- names[values != 0]

features_ridge <- train %>% select(names) %>% as.matrix()

# Retrain with selected variables
simple_ridge <- glmnet(features_ridge, target, family = 'binomial')


save(simple_ridge, file = 'Severity-Predictor/ridge.RData')
