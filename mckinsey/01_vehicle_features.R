library(tidyverse)



accidents <- read.csv('mckinsey/accidents.csv', stringsAsFactors = F)
test <- read.csv('mckinsey/test.csv', stringsAsFactors = F)
vehicles <- read.csv('mckinsey/vehicles.csv', stringsAsFactors = F)



head(vehicles)


maxes <- vehicles %>% 
  group_by(accident_id) %>% 
  summarise_if(is.numeric, max) %>% 
  rename_at(vars(2:8), function(x) paste0('max_', x))

mines <- vehicles %>% 
  group_by(accident_id) %>% 
  summarise_if(is.numeric, min) %>% 
  rename_at(vars(2:8), function(x) paste0('min_', x))

means <- vehicles %>% 
  group_by(accident_id) %>% 
  summarise_if(is.numeric, mean) %>% 
  rename_at(vars(2:8), function(x) paste0('mean_', x))

sums <- vehicles %>% 
  group_by(accident_id) %>% 
  summarise_if(is.numeric, sum) %>% 
  rename_at(vars(2:8), function(x) paste0('sum_', x))

num_vehicles_features <- vehicles %>% 
  count(accident_id) %>% 
  left_join(maxes) %>% 
  left_join(mines) %>% 
  left_join(means) %>% 
  left_join(sums)

head(vehicles)

# Split is temporal
max(accidents$date)
min(test$date)

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

modes <- vehicles %>% 
  group_by(accident_id) %>% 
  summarise_if(is.character, Mode)


modes <- modes %>% 
  rename_at(vars(2:15), function(x) paste0('mode_', x))


write.csv(modes, 'mckinsey/categorical_v1.csv', row.names = F)
write.csv(num_vehicles_features, 'mckinsey/numeric_v1.csv', row.names = F)
