
from datetime import datetime
import json
import logging
from airflow import DAG
import gspread
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import set_with_dataframe
from airflow.operators.python import PythonOperator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)

def download():
    logging.info("downloading...")

    SHEETS_ID = os.getenv('SHEETS_ID')
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(os.getenv('SHEETS_KEY')), ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID).worksheet("Modelowy")

    sheetValues = sheet.get_all_values()

    df = pd.DataFrame(sheetValues)
    df.columns = df.iloc[0]
    df = df[1:]
    logging.info("downloaded")

    df.to_csv('data.csv', index=False)


def dataPrep1():

    df = pd.read_csv('data.csv')

    logging.info("preparing part 1...")
    num_cols = ['person_age', 'person_income',
       'person_emp_exp', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score', 'loan_status']
    for col in num_cols:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)

    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        logging.info(f"Found {num_duplicates} duplicates. Removing them.")
        df = df.drop_duplicates()
    df.replace('', np.nan, inplace=True)

    logging.info("Finished prep part 1...")
    logging.info("saving file...")

    df.to_csv('data.csv', index=False)
    logging.info("saved file")

def dataPrep2():

    df = pd.read_csv('data.csv')

    scaler = StandardScaler()
    minMax = MinMaxScaler(feature_range=(0, 1))

    num_cols = ['person_age', 'person_income',
        'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'loan_status']

    df[num_cols] = scaler.fit_transform(df[num_cols])
    logging.info("Standardization done")
    
    df[num_cols] = minMax.fit_transform(df[num_cols])
    logging.info("normalization done")

    df.to_csv('data.csv', index=False)


def upload():
    

    df = pd.read_csv('data.csv')

    SHEETS_ID = os.getenv('SHEETS_ID')
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(os.getenv('SHEETS_KEY')), ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID)
    
    prepared = sheet.worksheet("Prepared")
    prepared.clear()

    set_with_dataframe(prepared, df)

    logging.info("Data successfully saved to Google Sheets")


with DAG(
    'prepare_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    download_task = PythonOperator(
        task_id='download_task',
        python_callable=download,
    )

    dataPrep1_task = PythonOperator(
        task_id='dataPrep1_task',
        python_callable=dataPrep1,
    )
    dataPrep2_task = PythonOperator(
        task_id='dataPrep2_task',
        python_callable=dataPrep2,
    )

    upload_task = PythonOperator(
        task_id='upload_task',
        python_callable=upload,
    )

    download_task >> dataPrep1_task >> dataPrep2_task >> upload_task
