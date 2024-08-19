## MLflow & Prefect

DAG Flow -

- We first load temperature data from Kaggle.
- This data is split to training data and validation data and the features are vectorized.
- 5 key ML models are used to find the model which has the lowest RMSE in validation data. LGBM regressor evolved as the best model. Hyperparameter tuning was
done using the search space built by Hyperopt python package.
- The best model will be train on all data to be ready to use in production.
- The dict vectorizer and best model are bundled into a pipeline and registered in the MLflow registry. All trained model artifacts are stored in AWS S3, identified by their run_id.
- The model deployed in Amazon S3 will be used in the deployment phase.

Change directory to training folder

cd ~/cyberbullying_detection/training
Run the following to start the mlflow tracking server

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root=s3://mlops-zoomcamp-cyberbullying/
Run prefect server

prefect server start
There are 2 options to run training pipeline.

Option 1: Run the prefect flow below

python training.py
Option 2: Trigger Prefect flow

Change directory to main folder

cd ~/cyberbullying_detection
Start the prefect worker

prefect worker start --pool cyberbullying
Deploy the pipeline

prefect deploy training/training.py:main_flow -n cyberbullying_flow -p cyberbullying
Run the ML model training and model registration pipeline

prefect deployment run 'Train Model Pipeline/cyberbullying_flow'
Below is the screenshot of the Prefect Deployment:


