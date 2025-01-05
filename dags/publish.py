from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
import os 

DOCKER_USERNAME = os.getenv('DOCKER_USERNAME')
DOCKER_PASSWORD = os.getenv('DOCKER_PASSWORD')

with DAG(
    'publish_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    build_image = BashOperator(
        task_id='build_docker_image',
        bash_command=f'docker build -t jaroslawgawrych/s25361_loan_approval_api:latest -f /opt/airflow/models/Dockerfile .',
        dag=dag,
    )

    docker_login = BashOperator(
        task_id='docker_login',
        bash_command=f"echo {DOCKER_PASSWORD} | docker login --username {DOCKER_USERNAME} --password-stdin",
        dag=dag,
    )

    publish_image = BashOperator(
        task_id='publish_docker_image',
        bash_command=f'docker push jaroslawgawrych/s25361_loan_approval_api:latest',
        dag=dag,
    )

    build_image >> publish_image << docker_login