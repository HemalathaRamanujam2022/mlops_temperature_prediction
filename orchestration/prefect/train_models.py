from typing import Callable, Dict, Optional, Tuple, Union

import pandas as pd
import pickle
from pathlib import Path
import os


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
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from prefect import flow, task, get_run_logger
from prefect_aws import S3Bucket, AwsCredentials
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
max_evaluations = 20
early_stopping_rounds = 1

HYPERPARAMETERS_WITH_CHOICE_INDEX = [
    'fit_intercept',
]


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

        mlflow.set_tag("model", model.__name__)
 
        model = model(**params) 

        print("Before fit : ")
        print("X_train : ", X_train.shape)
        print("X_val : ", X_val.shape)
        print("y_train : ", y_train.shape, type(y_train))
        print("y_val : ", y_val.shape, type(y_val))
        print("y_train : ", y_train[0:3])
        print("y_val : ", y_train[0:3])

        model.fit(X_train, y_train)
        
        print("After fit : ")

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # model_name_rmse = "rmse_" + type(model).__name__

        mlflow.log_metric("rmse" , rmse)
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
        lightbgm.LGBMRegressor
    """
    parts = module_and_class_name.split('.')

    if len(parts) > 1:
        cls = "sklearn"
        # print("parts : ", parts)
        # for part in parts:
            # cls = getattr(cls, part)

        module_submodule = cls + "." + parts[0]
        print("module_submodule : ", module_submodule)

        my_module = importlib.import_module(module_submodule)
        cls = getattr(my_module, parts[1])    
    else:
        cls = getattr(lightgbm, "LGBMRegressor") 

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

    search_space, choices = hp_space.build_hyperparamater_space(model_class, random_state)
  

    print("search_space : ", search_space)

    best_result = fmin(
        fn=partial(objective, X_train=X_train,X_val=X_val, model=model_class,
                   y_train=y_train,y_val=y_val,dv=dv),
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evaluations,
        trials=Trials()
    )

    # Convert choice index to choice value.
    for key in HYPERPARAMETERS_WITH_CHOICE_INDEX:
        if key in best_result and key in choices:
            idx = int(best_result[key])
            best_result[key] = choices[key][idx]

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
    print("X_train : ", X_train.shape, type(X_train))
    print("X_val : ", X_val.shape, type(X_val))
    print("y_train : ", y_train.shape, type(y_train))
    print("y_val : ", y_val.shape, type(y_val))
    print("y_train : ", y_train[0:3])
    print("y_val : ", y_val[0:3])

    return X_train, X_val, y_train, y_val, dv

@task(name="Train Models", log_prints=True)
def train_models(
    X_train: csr_matrix,
    X_val: csr_matrix,
    y_train: Series,
    y_val: Series,
    dv:DictVectorizer,  
    models_list: Series,
    max_evaluations: int
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
    

    for mdl_name in models_list:
                    # [ 
                    #   'ensemble.GradientBoostingRegressor',
                    #   'ensemble.RandomForestRegressor',
                    #   'linear_model.Lasso',
                    #   'linear_model.LinearRegression',
                    #   'svm.LinearSVR',
                    #   'LGBMRegressor'
                    # ]:
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


@task(name="Get Best Model", log_prints=True)
def get_best_model(client, EXPERIMENT_NAME):
    logger = get_run_logger()
    logger.info("Get the model with the highest test accuracy...")

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"])[0]
    print("best_run : ", best_run)
    best_run_id = best_run.info.run_id

    best_run_tags = best_run.data.tags
    print("best_run_tags : ", best_run_tags)
    tag_key = 'model'
    tag_value = best_run_tags.get(tag_key)
    
    return best_run_id, tag_value

@task(name="Re-train best model on all training data", log_prints=True)
def train_all_data(RUN_ID, X, y, tag_value):
    logger = get_run_logger()
    logger.info("Re-train best model on all data...")

    logged_model = f'runs:/{RUN_ID}/models_mlflow'
    model = mlflow.sklearn.load_model(logged_model)

    with mlflow.start_run() as run:
               
        mlflow.set_tag("developer", "hema")

        mlflow.log_param("input_file", "data/london_weather.csv")
     
        #Train the model
        # model.fit(X, y)

        # Log the model
        logger.info("Logging the model...")

        mlflow.set_tag("model", tag_value)
 
        mlflow.sklearn.log_model(model, "model")

        logger.info("Completed training process...")

        register_run_id = run.info.run_id

        return register_run_id

@task(name="Register Best Model", log_prints=True)
def register_best_model(client, register_run_id, model_name, tag_value):

    logger = get_run_logger()
    logger.info(f"Register the best model which has run_id: {register_run_id}...")

    result = mlflow.register_model(
        model_uri=f"runs:/{register_run_id}/models",
        name=model_name)
    
    # Add a description to the model version
    description = f'{tag_value} model retrained with all training data.'
    client.update_model_version(
        name=result.name,
        version=result.version,
        description=description
    )

    model_version = result.version
    new_stage = "Production"
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=False
    )

    logger.info(f"Model registered: {result.name}, version {result.version}...")

    return None

@task(name="Register Best Model", log_prints=True)
def export_model_to_s3_buckets(client, register_run_id):

    registered_run = client.get_run(
        run_id = register_run_id)
    print("registered_run : ", registered_run)

    artifact_uri = mlflow.get_run(register_run_id).info.artifact_uri
    print("artifact_uri : ", artifact_uri)

  
    # Load the model and the input file
    mlflow_model = artifact_uri

    experiment_id = mlflow.get_run(register_run_id).info.experiment_id

    local_folder = f"../../mlartifacts/{experiment_id}/{register_run_id}/artifacts"
    s3_folder = f"mlartifacts/{experiment_id}/{register_run_id}/artifacts"

    s3_bucket_block = S3Bucket.load("s3-bucket-block")
    s3_bucket_block.upload_from_folder(from_folder= local_folder, 
     to_folder=s3_folder)

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

    # # # Train the various models
    # models_list = [
    #                   'ensemble.GradientBoostingRegressor',
    #                   'ensemble.RandomForestRegressor',
    #                   'linear_model.Lasso',
    #                   'linear_model.LinearRegression',
    #                   'svm.LinearSVR',
    #                   'LGBMRegressor'
    #               ]
    # train_models(X_train, X_val, y_train, y_val, dv, models_list, max_evaluations=20)

    # # Get best model
    # best_run_id, tag_value = get_best_model(client, EXPERIMENT_NAME)

 

    # print("best_run_id : ", best_run_id)
    # print("tag_value : ", tag_value)
    # print("Selected model : ", [model for model in models_list if tag_value in model])

    # # Now run the best model by increasing max_evaluations to get a lower RMSE
    # best_model = [model for model in models_list if tag_value in model]
    # train_models(X_train, X_val, y_train, y_val, dv, best_model, max_evaluations=50)

    # # Get best model
    # best_run_id, tag_value = get_best_model(client, EXPERIMENT_NAME)

 

    # print("best_run_id : ", best_run_id)
    # print("tag_value : ", tag_value)
    # print("Selected model : ", [model for model in models_list if tag_value in model])

    # # # # Re-train best model on all data
    # # X_combined = np.r_[X_train, X_val]
    # # y_combined = np.r_[y_train, y_val]
    # X_combined = sp.vstack((X_train,X_val))
    # y_combined = pd.concat([y_train,y_val])
    # register_run_id = train_all_data(best_run_id, X_combined, y_combined, tag_value)
    # print("register_run_id : ", register_run_id)

    # # # Register best model
    # model_name = "London-Temperature-Prediction"
    # register_best_model(client, register_run_id, model_name, tag_value)

    # register_run_id = 'eca05e2b66b64d3fa1d927eff863216d'
    # export_model_to_s3_buckets(client, register_run_id)

    return None


if __name__ == "__main__":
    main_flow()
