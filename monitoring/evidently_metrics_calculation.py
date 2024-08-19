import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib
import math

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists evidently_metrics;
create table evidently_metrics(
	timestamp timestamp,
	reference_r2_score float,
	current_r2_score float,
    reference_rmse float,
    current_rmse float,
	reference_mean_error float,
    current_mean_error float,
    reference_mean_abs_error float,
    current_mean_abs_error float,
    reference_abs_error_max float,
    current_abs_error_max float
)
"""
with open('models/lgbm_reg.bin', 'rb') as f_in:
	lgbm_reg = joblib.load(f_in)

raw_data = pd.read_csv('../data/weather_data_2021.csv', parse_dates=True)
weather_features = ['month', 'max_temp', 'min_temp', 'global_radiation', 'sunshine', 'cloud_cover', 'snow_depth']

# raw_data.head()
raw_data = raw_data.drop("Unnamed: 0", axis=1)
raw_preds = lgbm_reg.predict(raw_data[weather_features])
raw_data['prediction'] = raw_preds
# raw_data.head()

# raw_data['date'].min()
# raw_data['date'].max()
col_mapping = ColumnMapping(target='mean_temp', prediction='prediction')
report = Report(metrics=[RegressionPreset()])
current_data = raw_data
reference_data = pd.read_parquet('data/reference.parquet')

weather_features = ['month', 'max_temp', 'min_temp', 'global_radiation', 'sunshine', 'cloud_cover', 'snow_depth']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=weather_features,
    target='mean_temp'
)

report = Report(metrics = [
      RegressionPreset()
])

@task(log_prints=True)
def prep_db():
	with psycopg.connect("host=localhost port=5433 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task(log_prints=True)
def calculate_metrics_postgresql(begin, curr, i):
    
    current = current_data.iloc[i * 10 : (i + 1) * 10]
    
    report.run(current_data=current, 
           reference_data=reference_data, 
           column_mapping=col_mapping
    )

    json_data = report.as_dict()
    print("json_data : ", json_data)

    # {'r2_score': 0.9766395601756793,
    #  'rmse': 0.8777466479934521,
    #  'mean_error': 0.0027740851855096314,
    #  'mean_abs_error': 0.6571490324589657,
    #  'mean_abs_perc_error': 5640013960804429.0,
    #  'abs_error_max': 9.045304773718234,
    #  'underperformance': {'majority': {'mean_error': 0.009526049060739545,
    #    'std_error': 0.6297230999966148},
    #   'underestimation': {'mean_error': -1.9537638567216289,
    #    'std_error': 0.5458481505374438},
    #   'overestimation': {'mean_error': 1.8382285804325214,
    #    'std_error': 0.8693058300255104}},
    #  'error_std': 0.877828959162133,
    #  'abs_error_std': 0.5819460599045371,
    #  'abs_perc_error_std': 565702135551166.4}

    ref_r2_score = json_data['metrics'][0]['result']['reference']['r2_score']
    cur_r2_score = json_data['metrics'][0]['result']['current']['r2_score']
    ref_rmse = json_data['metrics'][0]['result']['reference']['rmse']
    cur_rmse = json_data['metrics'][0]['result']['current']['rmse']
    ref_mean_error = json_data['metrics'][0]['result']['reference']['mean_error']
    cur_mean_error = json_data['metrics'][0]['result']['current']['mean_error']
    ref_mean_abs_error = json_data['metrics'][0]['result']['reference']['mean_abs_error']
    cur_mean_abs_error = json_data['metrics'][0]['result']['current']['mean_abs_error']
    ref_abs_error_max = json_data['metrics'][0]['result']['reference']['abs_error_max']
    cur_abs_error_max = json_data['metrics'][0]['result']['current']['abs_error_max']

    curr.execute(
        """
        INSERT INTO evidently_metrics (
            timestamp,
            reference_r2_score,
        	current_r2_score,
            reference_rmse,
            current_rmse,
        	reference_mean_error,
            current_mean_error,
            reference_mean_abs_error,
            current_mean_abs_error,
            reference_abs_error_max,
            current_abs_error_max
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            begin + datetime.timedelta(i),
            ref_r2_score,
        	cur_r2_score,
            ref_rmse,
            cur_rmse,
        	ref_mean_error,
            cur_mean_error,
            ref_mean_abs_error,
            cur_mean_abs_error,
            ref_abs_error_max,
            cur_abs_error_max
        ),
    )

@flow(log_prints=True)
def batch_monitoring():
    """
    Batch Monitoring Flow
    Prefect flow that orchestrates the monitoring process, including metric calculation and database insertion into Postgres DB.
        """
    SEND_TIMEOUT = 10
    prep_db()
    ROWS = current_data.shape[0]
    iters = math.ceil(ROWS / 10)
    begin = datetime.datetime.now(pytz.timezone('Asia/Kolkata')) - datetime.timedelta(iters)
    print("begin date time :  ", begin)
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example", autocommit=True) as conn:
        for i in range(iters):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(begin, curr, i)
    
            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")
 
if __name__ == '__main__':
	batch_monitoring()