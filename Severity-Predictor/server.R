#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {

    output$table <- renderTable({

        casualties <- input$casualties
        vehicles_inp <- paste0(sort(input$vehicles), collapse = '|')
        impact_inp <- paste0(sort(input$impact), collapse = '|')
        Vehicle_Manoeuvre_inp <- paste0(sort(input$Vehicle_Manoeuvre), collapse = '|')
        local_inp <- input$local
        
        load('ridge.RData')
        load('te.RData')
        
        veh_te_inp <- te_veh %>% filter(vehicles == vehicles_inp) %>% pull(vehicles_TargetEncoder)
        if(is.null(veh_te_inp)) veh_te_inp <- 0.17
        impact_te_inp <- te_impact %>% filter(impact_points == impact_inp) %>% pull(impact_points_TargetEncoder)
        if(is.null(impact_te_inp)) impact_te_inp <- 0.17
        local_te_inp <- te_local %>% filter(local_authority_district == local_inp) %>% pull(local_authority_district_TargetEncoder)
        if(is.null(local_te_inp)) local_te_inp <- 0.17
        maneu_te_inp <- te_maneu %>% filter(manoeuvres == Vehicle_Manoeuvre_inp) %>% pull(manoeuvres_TargetEncoder)
        if(is.null(maneu_te_inp)) maneu_te_inp <- 0.17
        
        moto <- as.integer(grepl('Moto|moto', vehicles_inp))
        if(is.null(moto)) moto <- 0
        ages <- as.integer(input$ages)
        mean_age <- mean((ages))
        
        simple_ridge$beta
        
        features <- as.matrix(c(local_te_inp, veh_te_inp, impact_te_inp, maneu_te_inp, casualties, mean_age, moto))
        if(length(features) != 7) features <- c(features, 0)
        dim(features) <- c(1, 7)
        
        print(features)
        prediction <- data.frame(
            severe_probability = as.vector(predict(simple_ridge, features, type = 'response', s = 0))
            ) %>% 
            mutate(severe_prediction = if_else(severe_probability < 0.2, 'Not severe', 'Severe'))
        
        prediction

    })
    
    output$plot <- renderPlot({
        
        casualties <- input$casualties
        vehicles_inp <- paste0(sort(input$vehicles), collapse = '|')
        impact_inp <- paste0(sort(input$impact), collapse = '|')
        Vehicle_Manoeuvre_inp <- paste0(sort(input$Vehicle_Manoeuvre), collapse = '|')
        local_inp <- input$local
        
        load('ridge.RData')
        load('te.RData')
        
        veh_te_inp <- te_veh %>% filter(vehicles == vehicles_inp) %>% pull(vehicles_TargetEncoder)
        if(is.null(veh_te_inp)) veh_te_inp <- 0.17
        impact_te_inp <- te_impact %>% filter(impact_points == impact_inp) %>% pull(impact_points_TargetEncoder)
        if(is.null(impact_te_inp)) impact_te_inp <- 0.17
        local_te_inp <- te_local %>% filter(local_authority_district == local_inp) %>% pull(local_authority_district_TargetEncoder)
        if(is.null(local_te_inp)) local_te_inp <- 0.17
        maneu_te_inp <- te_maneu %>% filter(manoeuvres == Vehicle_Manoeuvre_inp) %>% pull(manoeuvres_TargetEncoder)
        if(is.null(maneu_te_inp)) maneu_te_inp <- 0.17
        
        moto <- as.integer(grepl('Moto|moto', vehicles_inp))
        if(is.null(moto)) moto <- 0
        ages <- as.integer(input$ages)
        mean_age <- mean((ages))
        
        simple_ridge$beta
        
        features <- (c(local_te_inp, veh_te_inp, impact_te_inp, maneu_te_inp, casualties, mean_age, moto))
        if(length(features) != 7) features <- c(features, 0)
        
        coefs <- coef.glmnet(simple_ridge, s = 0)[-1]
        names_coef <- rownames(coef.glmnet(simple_ridge, s = 0))[-1]
        
        plot_df <- data.frame(feature = names_coef, contribution = coefs*features) %>% 
            mutate(Sign = contribution >= 0)
        
        print(plot_df)
        print(features)
        
        plot_df %>% 
            ggplot(aes(y = contribution, x = reorder(feature, contribution))) + 
            geom_bar(colour="black",stat = 'identity') + xlab("")+
            coord_flip() + theme_minimal() + ggtitle('Contribution of several variables')
        
    })

})
