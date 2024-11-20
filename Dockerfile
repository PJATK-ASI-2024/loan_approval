FROM python:3.12-slim
WORKDIR  /opt/airflow
RUN python -m pip install --upgrade pip setuptools wheel
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY .env ./
COPY dags ./dags