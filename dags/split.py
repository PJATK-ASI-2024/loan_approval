from datetime import datetime
import json
import logging
from airflow import DAG
import gspread
import os
import shutil
from google.oauth2.service_account import Credentials

import pandas as pd
from sklearn.model_selection import train_test_split
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
    path = kagglehub.dataset_download("taweilo/loan-approval-classification-data", force_download=True)
    logging.info(f"Downloaded dataset to {path}")

    for file_name in os.listdir(path):
        if os.path.exists(f"./{file_name}"):
            os.remove(f"./{file_name}")
        shutil.move(os.path.join(path, file_name), "./")

def split_data():

    src = pd.read_csv("./loan_data.csv")
    data70, data30 = train_test_split(src, test_size=0.3, random_state=80)

    logging.info(f"Total records: {src.shape[0]}")
    logging.info(f"Modelowy (70%): {data70.shape[0]}")
    logging.info(f"Douczeniowy (30%): {data30.shape[0]}")

    data70.to_csv('data70.csv', index=False)
    data30.to_csv('data30.csv', index=False)


def upload():
    SHEETS_ID = os.getenv('SHEETS_ID')

    credentials = Credentials.from_service_account_info(json.loads(os.getenv('SHEETS_KEY')), scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    data30 = pd.read_csv('data30.csv')
    data70 = pd.read_csv('data70.csv')

    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID)
    
    modelowy = sheet.worksheet("Modelowy")
    modelowy.clear()
    set_with_dataframe(modelowy, data70)

    douczeniowy = sheet.worksheet("Douczeniowy")
    douczeniowy.clear()
    set_with_dataframe(douczeniowy, data30)

    logging.info("Data successfully saved to Google Sheets")

with DAG(
    'split_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    download_task = PythonOperator(
        task_id='download_task',
        python_callable=download,
    )

    split_task = PythonOperator(
        task_id='split_task',
        python_callable=split_data,
    )

    upload_task = PythonOperator(
        task_id='upload_task',
        python_callable=upload,
    )

    download_task >> split_task >> upload_task

