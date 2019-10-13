This folder contains all the scripts, analysis and figures that have been use for the Restb.ai challenge at HackUPC19.

Our proposal is an algorithm that uses the Restb.ai APIs for real estate to estimate the fair value price of a specific room in Airbnb by evaluating only just one photo of the room.

The model is a Gradient Boosting that has been trained by a dataset of prices and the variables given by the Restb.ai APIs by giving them some real images. The training, and validation, dataset consisting in photos and prizes has been scrapped out from Airbnb spanish website using Selenium. More specifically, the data has been queried from the firsts results of the Airbnb search algorithm under the following restrictions.

- All kind of rooms
- In all the city of Barcelona
- During the 22nd and 24th of November 2019. (Looking into one month in advance during weekdays so as to impose a better estimation of the fair price)

It also contains a Python notebook that contains the main algorithm that can be divided into:

-Loading scrapped dataset
-Analyze images through Restb.ai API
-Construct training table
-Crossvalidate and validate the XGB model
-Predict a given image.

Improvements of the model:
-Scrap more than one image of each Airbnb (difficult task) so as to have better features and information about the kitchen and the bathroom, so as to use the conditions APIs of Restb.ai
-Ask the user for more data such as the zip code, the squared meters, the number of bathrooms, ... That will point directly to our target.
