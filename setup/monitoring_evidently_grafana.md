### Model monitoring

Evidently AI is a tool that is used to track the perfrmance metrics of a model over time. There are several metrics like Regression Presets, Data Quaity Drift, Data and Target drifts and Regression Performance metrics that are useful to track to monitor the performance of a model. Evindently AI can generate reprts and dashboards on these metrics which can be accessed using their UI. The Evidently and Grafana tools are run inside a docker container.

To start the service, run the command
```
docker compose -up --build
```
Evidently UI can be accessed on port 8000 on the localhost using the command.

```
evidently ui
```
![Rep1](../images/Evidently_Dashboard.png)  
![Rep2](../images/Evidently_Reports.png)  


The evidently AI also stores the metrics in a Postgres DB which can  be visualised using Grafana reporting tool. The SQL inserts are orchestrated via a Prefect flow.

New data will be imported amd the model will be run on this data. The original data used to build the validation set for the model will serve as the reference data and the recent data as the current data. If the quality checks and data drift are not within acceptable thresholds on the new data, the model will need to be tuned on the new data again.


First, change directory to [monitoring folder](../monitoring)
```
cd ~/mlops_teperature_prediction/monitoring
```

Run the [baseline notebook](../monitoring/baseline_temperature_data.ipynb) that builds the reference data and also builds some basic reports on the Evidently AI tool.

Generate evidently report and tracking database for grafana monitoring dashboard by calling the script - 
```
python evidently_metrics_calculation.py
```

You can view the Grafana dashboard on the following URL. Use the username "admin" and password "admin" for Grafana.
```
http://localhost:3000
```

The dashboard will look like the following. The dashboard provides metrics such as r2_Score, rmse, mean absolute error and absolute error max.

![Grafana](images/Grafana_Metrics_Dashboard.png)


