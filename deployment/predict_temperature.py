import os
import pickle
import pandas as pd
import numpy as np

import mlflow
from flask import Flask, request, jsonify

import boto3
from botocore.exceptions import ClientError

import lightgbm

# Enable logging
# import logging
# logging.basicConfig(level=logging.DEBUG)



RUN_ID = os.getenv('RUN_ID')
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
AWS_DEFAULT_REGION = os.getenv("AWS_REGION")

def verify_credentials():
    
    s3 = boto3.setup_default_session(region_name=aws_region)
    s3 = boto3.session.Session().client("s3", 
                                        aws_access_key_id = aws_access_key,
                                        aws_secret_access_key = aws_secret_access_key,)
    # s3 = boto3.client('s3',
    # aws_access_key_id = aws_access_key,
    # aws_secret_access_key = aws_secret_access_key,
    # region_name = aws_region
    # )
    try:
        bucket_response = s3.list_buckets()
        # Output the bucket names
        print('Existing buckets are:')
        for bucket in bucket_response ['Buckets']:
            print(f' {bucket["Name"]}') 
    except ClientError:
        print("Credentials are NOT valid.")


def prepare_features(weather_data):

    features = {}

    df = pd.DataFrame(weather_data, index=[0])
    print(df)

    df['date'] = pd.to_datetime(df["recorded_date"],format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df["date"].dt.month.map("{:02}".format)

    weather_features = ['month', 'max_temp', 'min_temp', 'global_radiation', 'sunshine', 'cloud_cover', 'snow_depth']

    features = df[weather_features].to_dict(orient='records')
    nd_array = df[weather_features].values
    print("features : ", features, type(features), nd_array)
    return features


def predict(features):

    s3 = boto3.setup_default_session(region_name=aws_region)
    s3 = boto3.session.Session().client("s3", 
                                        aws_access_key_id = aws_access_key,
                                        aws_secret_access_key = aws_secret_access_key,)

    try:
        logged_model = f's3://mlops-temperature-prediction/mlartifacts/1/{RUN_ID}/artifacts/models_mlflow'
        # logged_model = f'runs:/{RUN_ID}/model'
        print("logged_model : ", logged_model)
        model = mlflow.pyfunc.load_model(logged_model)

        preds = model.predict(features)
        return float(preds[0])
    except ClientError:
        print("Credentials are NOT valid.")


app = Flask('temperature-prediction')


@app.route('/predict_temperature', methods=['POST'])
def predict_endpoint():
   
    weather_data = request.get_json()
    verify_credentials()
    features_matrix = prepare_features(weather_data)
    print(features_matrix)
    pred = predict(features_matrix)

    result = {
        'mean_temperature': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)