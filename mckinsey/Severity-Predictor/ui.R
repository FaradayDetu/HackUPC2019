#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(glmnet)
library(dplyr)
library(ggplot2)

accidents <- read.csv('accidents.csv', stringsAsFactors = F)
vehicles <- read.csv('vehicles.csv', stringsAsFactors = F)
load('ridge.RData')
load('te.RData')


# Define UI for application that draws a histogram
shinyUI(fluidPage(

    # Application title
    titlePanel("Serverity Predictor"),

    # Sidebar with a slider input for number of bins
    sidebarLayout(
        
        sidebarPanel(
            sliderInput("casualties",
                        "Casualties:",
                        min = -1,
                        max = 60,
                        value = 0),
            selectInput(inputId = "vehicles",
                        label="Vehicles in accident: ",
                        choices = unique(vehicles$Vehicle_Type), 
                        selected = "Car",
                        multiple = T,
                        selectize = TRUE, 
                        width = NULL, 
                        size = NULL),
            selectInput(inputId = "impact",
                        label="Points of impact: ",
                        choices = unique(vehicles$X1st_Point_of_Impact), 
                        selected = "Front",
                        multiple = T,
                        selectize = TRUE, 
                        width = NULL, 
                        size = NULL),
            selectInput(inputId = "Vehicle_Manoeuvre",
                        label="Vehicle Manoeuvre: ",
                        choices = unique(vehicles$Vehicle_Manoeuvre), 
                        selected = "Turning right",
                        multiple = TRUE,
                        selectize = TRUE, 
                        width = NULL, 
                        size = NULL),
            selectInput(inputId = "local",
                        label="Local District: ",
                        choices = unique(accidents$local_authority_district), 
                        selected = "London",
                        multiple = F,
                        selectize = TRUE, 
                        width = NULL, 
                        size = NULL),
            selectInput(inputId = "ages",
                        label="Age of people: ",
                        choices = 1:100, 
                        selected = 30,
                        multiple = TRUE,
                        selectize = TRUE, 
                        width = NULL, 
                        size = NULL)
   
        ),
        
        # Show a plot of the generated distribution
        mainPanel(
            tableOutput("table"), 
            plotOutput("plot")
        )
    )
))
