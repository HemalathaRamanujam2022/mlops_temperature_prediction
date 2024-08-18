## <ins>Temperature prediction in London</ins>
This project is used to predict the mean temperature in London on a given date based on important weather parameters gathered by European Climate Assessment & Dataset project.

### Problem
The mean temperature at a location has a close connection to climate change and has a wide-ranging impact on the environment, society and global economies. The global mean temperatures have been steadily increasing due to the emission of green house gases. Human kind is increasingly exposed to extreme weather events like heatwaves, hurricanes and flooding. The rising sea levels can lead to habitat loss. Elevated temperatures worsen air quality thereby posing a serious health hazard to vulnerable populations. Increased temperatures have a profound effect on agricultural productivity and can lead to water scarcity stressing out the natutal and man made water resources. This project aims at creating models that can predict the mean temperature accurately such that proactive measures can be undertaken to maintain balance between environment, agricultural demands, energy management and health and safety areas.

### Objective
This project creates a machine learning (ML) operations pipeline that will use the weather data to predict mean temperature in London. Several ML models will be run on this data and the experiment will be tracked using MLflow and best model logged into MLflow Registry. The best model’s artifacts will be stored on AWS S3 and accessed inside a Flask application as a web service to make the mean temperature predictions. The entire pipeline from data preparation, feature engineering, model training, hyper parameter tuning to model tracking and storage will be orchestrated using Prefect. For monitoring the performance of the model on new weather data, Evidently AI and Grafana tools are used.

### Dataset
The data for this project is downloaded from Kaggle <https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data> . The dataset is created by “European Climate Assessment & Dataset project” and has around 15K records for the time period from 1979 to 2021. The data is captured daily.


Source URL : <https://smoosavi.org/datasets/us_accidents> 

<https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents>
