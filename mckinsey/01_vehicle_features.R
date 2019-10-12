
# This script is to generate features using the vehicles table
# The difficult thing is that there are many vehicles per accident so we have
# to compute aggregates of features

# Load libraries and data
library(tidyverse)

accidents <- read.csv('mckinsey/accidents.csv', stringsAsFactors = F)
test <- read.csv('mckinsey/test.csv', stringsAsFactors = F)
vehicles <- read.csv('mckinsey/vehicles.csv', stringsAsFactors = F)

# We agregate each numeric column using max,min,mean and sum
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

# We join all the information via accident_id
num_vehicles_features <- vehicles %>% 
  count(accident_id) %>% 
  left_join(maxes) %>% 
  left_join(mines) %>% 
  left_join(means) %>% 
  left_join(sums)

# We aggregate categorical features using the mode. 
# This is a first step and we perform better aggregation after.
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

modes <- vehicles %>% 
  group_by(accident_id) %>% 
  summarise_if(is.character, Mode)


modes <- modes %>% 
  rename_at(vars(2:15), function(x) paste0('mode_', x))

# Write output files
write.csv(modes, 'mckinsey/categorical_v1.csv', row.names = F)
write.csv(num_vehicles_features, 'mckinsey/numeric_v1.csv', row.names = F)

# We aggregate vehicles via pasting. 
# It is fundamental to arrange by vehicle type
vehicles %>% 
  arrange(Vehicle_Type) %>% 
  group_by(accident_id) %>% 
  summarise(vehicles = paste0(Vehicle_Type, collapse = '|')) %>% 
  write.csv('mckinsey/categorical_v2.csv', row.names = F)

# We created motorcycle features
vehicles %>% 
  arrange(Vehicle_Type) %>% 
  group_by(accident_id) %>% 
  summarise(vehicles = paste0(Vehicle_Type, collapse = '|')) %>% 
  mutate(lower_vehicle = tolower(vehicles),
         moto_in_accident = as.integer(grepl('moto', lower_vehicle))) %>% 
  write.csv('mckinsey/categorical_v2-1.csv', row.names = F)

# Same with point of impact and manouver
vehicles %>% 
  arrange(X1st_Point_of_Impact) %>% 
  group_by(accident_id) %>% 
  summarise(impact_points = paste0(X1st_Point_of_Impact, collapse = '|')) %>% 
  write.csv('mckinsey/categorical_v3.csv', row.names = F)


vehicles %>% 
  arrange(Vehicle_Manoeuvre) %>% 
  group_by(accident_id) %>% 
  summarise(manoeuvres = paste0(Vehicle_Manoeuvre, collapse = '|')) %>% 
  write.csv('mckinsey/categorical_v4.csv', row.names = F)



