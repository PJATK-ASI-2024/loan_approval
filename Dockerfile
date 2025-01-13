FROM apache/airflow:2.10.3
ADD requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt
RUN pip install --no-cache-dir apache-airflow-providers-docker==2.1.0
USER root
RUN echo "airflow ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER airflow
