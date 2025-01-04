FROM apache/airflow:2.10.3 AS airflow-stage
ADD requirements.txt .
RUN pip install --no-cache-dir apache-airflow==${AIRFLOW_VERSION} apache-airflow-providers-docker -r requirements.txt
COPY .env .
RUN echo "source ./.env" >> ~/.bashrc

FROM python:3.12 AS api-stage
WORKDIR /app
RUN pip install --upgrade pip
ADD requirements.txt .
RUN pip install -r requirements.txt 
RUN pip install apache-airflow-providers-docker
COPY ./API/ .
COPY ./models ./models
EXPOSE 5000
CMD ["uvicorn", "server:app", "--reload", "--port=5000", "--host=0.0.0.0"]