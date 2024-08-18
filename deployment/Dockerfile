FROM python:3.9.19-slim

# ENV RUN_ID=${RUN_ID}
# ENV AWS_ACCESS_KEY=${AWS_ACCESS_KEY}
# ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict_temperature.py", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict_temperature:app" ]