FROM apache/airflow:2.10.3
ADD requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt
COPY .env .
RUN echo "source ./.env" >> ~/.bashrc

FROM python:3.12
WORKDIR /app
COPY . ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["uvicorn", "server:app", "--reload", "--port=8080", "--host=0.0.0.0"]