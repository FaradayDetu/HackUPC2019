# library(tidyverse)

casualties <- 10
vehicles_inp <- 'Car|Car'

veh_te_inp <- te_veh %>% filter(vehicles == vehicles_inp) %>% pull(vehicles_TargetEncoder)
impact_te_inp <- te_impact %>% filter(impact_points == impact_inp) %>% pull(impact_points_TargetEncoder)
impact_te_inp <- te_local %>% filter(local_authority_district == local_inp) %>% pull(local_authority_district_TargetEncoder)
maneu_te_inp <- te_maneu %>% filter(manoeuvres == Vehicle_Manoeuvre_inp) %>% pull(manoeuvres_TargetEncoder)
