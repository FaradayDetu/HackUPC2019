# McKinsey challenge HackUPC2019 


Random Forest Gump team solution consists on heavy feature engineering, deep insights on the dataset
and deployment of a solution to help emergency services. 


The best model has been chosen by using time-series validation. This makes sense as the data
has been split in a temporal manner and because we want to predict the severeness of future accidents. One fundamental idea to improve the scores of our models has been chosing an optimal threshold on the training data based on F1 score. After that, advanced featuring engineering has been applied to our data set. The ideas that helped the most were agreggating vehicles information and applying target encoding to cathegorical variables. Our best model is a catboost that scores 0.5 in train and 0.44 in validation.

Additionally, we have developed a shiny app, using a simpler but performant model (lasso) that allows us, based on a small set of features, to predict if an accident is severe or not. Having few features allows us to introduce data manually by a customer service agent. The model is also interpretable and the shiny app shows the contribution of each variable to the final score. 