
from datetime import datetime
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
    'publish_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
    ) as dag:

    build_docker_image = DockerOperator(
        task_id='build_docker_image',
        image="docker:latest",
        api_version='auto',
        command=f"docker build -t jaroslawgawrych/s25361_loan_approval_api:latest -f ./Dockerfile ./",
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge', 
        dag=dag,
    )

    publish_docker_image = DockerOperator(
        task_id='publish_docker_image',
        image="docker:latest", 
        api_version='auto',
        command=f"docker push jaroslawgawrych/s25361_loan_approval_api:latest",
        docker_url='unix://var/run/docker.sock',
        network_mode='bridge',
        dag=dag,
    )

    build_docker_image >> publish_docker_image