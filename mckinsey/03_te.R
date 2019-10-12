# Train simple linear model

library(tidyverse)


te <- read.csv('mckinsey/david_happy.csv')

# Save target encoding results
te_veh <- te %>% count(vehicles, vehicles_TargetEncoder)
te_impact <- te %>% count(impact_points, impact_points_TargetEncoder)
te_maneu <- te %>% count(manoeuvres, manoeuvres_TargetEncoder)
te_local <- te %>% count(local_authority_district, local_authority_district_TargetEncoder)

save(te_veh, te_impact, te_maneu, te_local, file = 'Severity-Predictor/te.RData')


