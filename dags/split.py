from datetime import datetime
import json
import logging
from airflow import DAG
import gspread
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from oauth2client.service_account import ServiceAccountCredentials
from airflow.operators.python import PythonOperator
from gspread_dataframe import set_with_dataframe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)

def download():
    import kagglehub
    path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")
    logging.info(f"Downloaded dataset to {path}")

    for file_name in os.listdir(path):
        if os.path.exists(f"./{file_name}"):
            os.remove(f"./{file_name}")
        shutil.move(os.path.join(path, file_name), "./")

def split_data(**kwargs):

    src = pd.read_csv("./loan_data.csv")
    data70, data30 = train_test_split(src, test_size=0.3, random_state=80)

    logging.info(f"Total records: {src.shape[0]}")
    logging.info(f"Modelowy (70%): {data70.shape[0]}")
    logging.info(f"Douczeniowy (30%): {data30.shape[0]}")

    data70.to_csv('/tmp/data70.csv', index=False)
    data30.to_csv('/tmp/data30.csv', index=False)

    kwargs['ti'].xcom_push(key='data70_path', value='/tmp/data70.csv')
    kwargs['ti'].xcom_push(key='data30_path', value='/tmp/data30.csv')


def upload(**kwargs):
    SHEETS_ID = os.getenv('SHEETS_ID')

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(os.getenv('SHEETS_KEY')), ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID)
    
    data70_path = kwargs['ti'].xcom_pull(task_ids='split_data', key='data70_path')
    data30_path = kwargs['ti'].xcom_pull(task_ids='split_data', key='data30_path')

    data70 = pd.read_csv(data70_path)
    data30 = pd.read_csv(data30_path)

    modelowy = sheet.worksheet("Modelowy")
    modelowy.clear()
    set_with_dataframe(modelowy, data70)

    douczeniowy = sheet.worksheet("Douczeniowy")
    douczeniowy.clear()
    set_with_dataframe(douczeniowy, data30)

    logging.info("Data successfully saved to Google Sheets")

with DAG(
    'split',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
) as dag:

    download_data = PythonOperator(
        task_id='download_data',
        python_callable=download,
    )

    split_data_task = PythonOperator(
        task_id='split_data',
        python_callable=split_data,
        provide_context=True
    )

    save_data_to_sheets = PythonOperator(
        task_id='upload_data',
        python_callable=upload,
        provide_context=True
    )

    download_data >> split_data_task >> save_data_to_sheets
