import requests

weather_data = {
    "recorded_date" : "19790101",	
    "cloud_cover" : 2	,
    "sunshine" : 7,
    "global_radiation" : 52,
    "max_temp" : 2.3,	
    "min_temp" : -7.5,	
    "precipitation" : 0.4,
    "snow_depth" : 9.0
    }

# weather_data = {
#     "recorded_date" : "19790102",	
#     "cloud_cover" : 6	,
#     "sunshine" : 1.7,
#     "global_radiation" : 27,
#     "max_temp" : 1.6,	
#     "min_temp" : -7.5,	
#     "precipitation" : 0,
#     "snow_depth" : 8.0
#     }

url = 'http://localhost:9696/predict_temperature'
response = requests.post(url, json=weather_data)
print(response.json())