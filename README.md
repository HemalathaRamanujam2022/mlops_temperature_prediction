## <ins>Temperature prediction in London</ins>
This is an end to end project to implement a machine learning operations pipeline that predicts the mean temperature in London on a given date based on important weather parameters gathered by European Climate Assessment & Dataset project. This project is implemented using MLflow, Prefect, Amazon Web Services (EC2, S3, IAM), Docker, Jupyter Notebook, Flask, Docker, Evidently and Grafana. 

## Description 

### Problem
The mean temperature at a location has a close connection to climate change and has a wide-ranging impact on the environment, society and global economies. The global mean temperatures have been steadily increasing due to the emission of green house gases. Human kind is increasingly exposed to extreme weather events like heatwaves, hurricanes and flooding. The rising sea levels can lead to habitat loss. Elevated temperatures worsen air quality thereby posing a serious health hazard to vulnerable populations. Increased temperatures have a profound effect on agricultural productivity and can lead to water scarcity stressing out the natutal and man made water resources. This project aims at creating models that can predict the mean temperature accurately such that proactive measures can be undertaken to maintain balance between environment, agricultural demands, energy management and health and safety areas.

### Objective
This project creates a machine learning (ML) operations pipeline that will use the weather data to predict mean temperature in London. Several ML models will be run on this data and the experiment will be tracked using MLflow and best model logged into MLflow Registry. The best model’s artifacts will be stored on AWS S3 and accessed inside a Flask application as a web service to make the mean temperature predictions. The entire pipeline from data preparation, feature engineering, model training, hyper parameter tuning to model tracking and storage will be orchestrated using Prefect. For monitoring the performance of the model on new weather data, Evidently AI and Grafana tools are used.

### Dataset
The data for this project is downloaded from [Kaggle](https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data) . The dataset is created by “European Climate Assessment & Dataset project” and has around 15K records for the time period from 1979 to 2021. The data is captured daily. The data card on the above link can be obtained from the above link.

### Tools and Technologies Used

Cloud – [Amazon Web Services](aws.amazon.com)
Virtual machine – [Amazon EC2](ec2.amazon.com)
Containerization – [Docker and Docker Compose](https://www.docker.com/)
Orchestration – [Prefect](https://www.prefect.io/)
Experiment tracking and ML model management – [Mlflow](https://mlflow.org/)
Model artifacts storage – [Amazon S3](https://aws.amazon.com/s3/)
Model deployment – [Flask](https://flask.palletsprojects.com/en/3.0.x/)
Model monitoring – [Evidently AI](https://www.evidentlyai.com/)
Model metrics visualisation - [Grafana](https://grafana.com/) 
Language – [Python](https://www.python.org/)
Notebooks – [Jupyter](https://jupyter.org/)


