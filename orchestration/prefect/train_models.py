from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd
import pickle
from pathlib import Path


import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import sklearn
from sklearn.base import BaseEstimator
from pandas import Series
from scipy.sparse._csr import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger

import hp_space
# from hp_space import build_hyperparamater_space 

from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import importlib

# Global variables
numeric_features =  ['max_temp', 'min_temp', 'mean_temp', 'global_radiation', 'sunshine', 'cloud_cover', 'snow_depth', 'precipitation', 'pressure']
weather_features = ['month', 'max_temp', 'min_temp', 'global_radiation', 'sunshine', 'cloud_cover', 'snow_depth']
weather_target = "mean_temp"
input_file = "data/london_weather.csv"
random_state = 42
max_evaluations = 1
early_stopping_rounds = 1

@task(name="Load Data", log_prints=True, retries=3, retry_delay_seconds=2)
def load_data(path):
    """
    Load Data from CSV File
    Load the data from the specified CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    logger = get_run_logger()
    logger.info("Loading data from %s", path)
    df = pd.read_csv(input_file, parse_dates=True)
    return df

def preprocess_data(df):
    """
    Split the date field into month and year. Assign mean values to all NaN
    """
    df['date'] = pd.to_datetime(df["date"],format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df["date"].dt.month.map("{:02}".format)
    weather_all_features = df.drop(["mean_temp", "date"], axis=1).columns.to_list()
    [df[col].fillna(df[col].mean(), inplace=True) for col in numeric_features]
    return df

def objective(params, X_train, X_val, model, y_train, y_val, dv):

    print("params passed to objective function : ", params)

    with mlflow.start_run():
    
        mlflow.set_tag("developer", "hema")

        mlflow.log_param("train-data-path", "data/london_weather.csv")
        mlflow.log_param("test-data-path","data/london_weather.csv")

        mlflow.log_param("model", model)

        model = model(**params) 

        print("Before fit : ")
        print("X_train : ", X_train.shape)
        print("X_val : ", X_val.shape)
        print("y_train : ", y_train.shape)
        print("y_val : ", y_val.shape)

        model.fit(X_train, y_train)
        
        print("After fit : ")

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        model_name_rmse = "rmse_" + type(model).__name__

        mlflow.log_metric(model_name_rmse , rmse)
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")
        Path("models").mkdir(parents=True, exist_ok=True)
        with open('models/preprocessor.b', 'wb') as f_out:
            pickle.dump((dv, model), f_out)
        mlflow.log_artifact(local_path="models/preprocessor.b", artifact_path="preprocessor")

        return {'loss': rmse, 'status': STATUS_OK}


@task(name="Prepare Data", log_prints=True)
def prepare_data(df):
    """

    Preprocess the weather data
    Args:   
        pd.DataFrame: Dataframe with weather information

    Returns:
        X_train, y_train: data which ready to be use
    """
    logger = get_run_logger()
    logger.info("Preprocessing data: Started")

    df = preprocess_data(df)
    X = df[weather_features]
    y = df[weather_target]

    logger.info("Preprocessing data: Completed")

    return X, y

@task(name="Load Model Class", log_prints=True)
def load_class(module_and_class_name: str) -> BaseEstimator:
    """
        module_and_class_name:
        ensemble.ExtraTreesRegressor
        ensemble.GradientBoostingRegressor
        ensemble.RandomForestRegressor
        linear_model.Lasso
        linear_model.LinearRegression
        svm.LinearSVR
    """
    parts = module_and_class_name.split('.')
    cls = "sklearn"
    # print("parts : ", parts)
    # for part in parts:
        # cls = getattr(cls, part)

    module_submodule = cls + "." + parts[0]
    print("module_submodule : ", module_submodule)

    my_module = importlib.import_module(module_submodule)
    cls = getattr(my_module, parts[1])    

    print("The model class is : ", cls)
    return cls

@task(name="Tune Hyperparameters", log_prints=True)
def tune_hyperparameters(
                        X_train: csr_matrix,
                        y_train: Series,
                        model_class: BaseEstimator,
                        X_val: csr_matrix,
                        y_val: Series,
                        dv:DictVectorizer,
                        # hyperparameters: Optional[Dict] = None,
                        max_evaluations: int = max_evaluations,
                        random_state: int = random_state,
                        ):
    
    print("Inside tune_hyperparameters() function")
    print("model_class : ", model_class)

    search_space = hp_space.build_hyperparamater_space(model_class, random_state)

    print("search_space : ", search_space)

    best_result = fmin(
        fn=partial(objective, X_train=X_train,X_val=X_val, model=model_class,
                   y_train=y_train,y_val=y_val,dv=dv),
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evaluations,
        trials=Trials()
    )

    print("The best result for model {model_class} is :, best_result")

    return best_result

@task(name="Vectorize Features", log_prints=True)
def vectorize_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> tuple(
    [
        csr_matrix,
        csr_matrix,
        Series,
        Series,
        sklearn.feature_extraction.DictVectorizer,
    ]
):
    """Standardize and Vectorize the features"""
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(df_train, df_val, test_size=0.33, random_state=random_state)

    dv = DictVectorizer()
    X_train = dv.fit_transform(X_train.to_dict(orient='records'))
    X_val  = dv.transform(X_val.to_dict(orient='records'))
    # Scale the data
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val) 

    print("Vectorize features : ")
    print("X_train : ", X_train.shape)
    print("X_val : ", X_val.shape)
    print("y_train : ", y_train.shape)
    print("y_val : ", y_val.shape)

    return X_train, X_val, y_train, y_val, dv

@task(name="Train Models", log_prints=True)
def train_models(
    X_train: csr_matrix,
    X_val: csr_matrix,
    y_train: Series,
    y_val: Series,
    dv:DictVectorizer,  
    ):
    """
    Train Model
    Args:
        X, y: features and target data which ready to be use

    Returns:
        machine learning model which is saved in defined directory
    """
    logger = get_run_logger()
    logger.info("Starting Linear Regression training process...")
    

    for mdl_name in ['linear_model.Lasso']:
        model_class = load_class(mdl_name)
        model_instance = model_class()
        print("The loaded model class is : ", type(model_class))
        print(" ", type(model_instance), model_instance)

        result = tune_hyperparameters(
                            X_train=X_train,
                            y_train=y_train,
                            model_class = model_class,
                            X_val=X_val,
                            y_val=y_val,
                            dv=dv,
                            max_evaluations=max_evaluations,
                            random_state=random_state,
                    )
    logger.info("Completed training process...")

    return None


# @task(name="Get Best Model", log_prints=True)
# def get_best_model(client, EXPERIMENT_NAME):
#     logger = get_run_logger()
#     logger.info("Get the model with the highest test accuracy...")

#     experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
#     best_run = client.search_runs(
#         experiment_ids=experiment.experiment_id,
#         run_view_type=ViewType.ACTIVE_ONLY,
#         max_results=1,
#         order_by=["metrics.test_accuracy DESC"])[0]
#     best_run_id = best_run.info.run_id

#     best_run_tags = best_run.data.tags
#     tag_key = 'model'
#     tag_value = best_run_tags.get(tag_key)

#     return best_run_id, tag_value

# @task(name="Re-train best model on all training data", log_prints=True)
# def train_all_data(S3_BUCKET_NAME, RUN_ID, X, y, tag_value):
#     logger = get_run_logger()
#     logger.info("Re-train best model on all data...")

#     logged_model = f's3://{S3_BUCKET_NAME}/1/{RUN_ID}/artifacts/model'
#     model = mlflow.sklearn.load_model(logged_model)

#     with mlflow.start_run() as run:
        
#         #Train the model
#         model.fit(X, y)

#         # Log the model
#         logger.info("Logging the model...")

#         mlflow.set_tag("model", tag_value)
#         mlflow.sklearn.log_model(model, "model")

#         logger.info("Completed training process...")

#         register_run_id = run.info.run_id

#         return register_run_id

# @task(name="Register Best Model", log_prints=True)
# def register_best_model(client, register_run_id, model_name, tag_value):

#     logger = get_run_logger()
#     logger.info(f"Register the best model which has run_id: {register_run_id}...")

#     result = mlflow.register_model(
#         model_uri=f"runs:/{register_run_id}/models",
#         name=model_name)
    
#     # Add a description to the model version
#     description = f'{tag_value} model retrained with all training data.'
#     client.update_model_version(
#         name=result.name,
#         version=result.version,
#         description=description
#     )
#     logger.info(f"Model registered: {result.name}, version {result.version}...")

#     return None

@flow(name="Temperature Prediction Training Pipeline", log_prints=True)
def main_flow():
    """
    Train Model Flow
    Prefect flow that orchestrates the data loading, preprocessing, and model training process.
    """
    logger = get_run_logger()
    logger.info("Starting flow...")

    # MLflow settings
    # S3_BUCKET_NAME = "mlops-zoomcamp-cyberbullying"
    MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    EXPERIMENT_NAME = "mean-temperature-prediction-models"
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # Load the data
    df = load_data(input_file)
    print(df.head())

    # Clean the text
    df_train, df_val = prepare_data(df)
    print(df_train[0:2])
    print(df_val[0:2])

    # Vectorize the features
    X_train, X_val, y_train, y_val, dv = vectorize_features(df_train, df_val)

    # Train the various models
    train_models(X_train, X_val, y_train, y_val, dv)

    # # Get best model
    # best_run_id, tag_value = get_best_model(client, EXPERIMENT_NAME)

    # # Re-train best model on all data
    # register_run_id = train_all_data(S3_BUCKET_NAME, best_run_id, X_train, y_train, tag_value)

    # # Register best model
    # model_name = "Cyberbullying-classification"
    # register_best_model(client, register_run_id, model_name, tag_value)

    return None


if __name__ == "__main__":
    main_flow()
