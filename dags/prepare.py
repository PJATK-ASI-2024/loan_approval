
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

def download(**kwargs):
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

    temp_file = '/tmp/modelowy.csv'
    df.to_csv(temp_file, index=False)
    kwargs['ti'].xcom_push(key='download_path', value=temp_file)

def dataPrep1(**kwargs):
    logging.info("preparing part 1...")

    download_path = kwargs['ti'].xcom_pull(task_ids='download_data', key='download_path')
    df = pd.read_csv(download_path)

    num_cols = ['person_age', 'person_income',
       'person_emp_exp', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score', 'loan_status']
    for col in num_cols:
        df[col] = df[col].str.replace(',', '.', regex=False)

    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        logging.info(f"Found {num_duplicates} duplicates. Removing them.")
        df = df.drop_duplicates()
    df.replace('', np.nan, inplace=True)
    return df

def dataPrep2(**kwargs):

    prep1_path = kwargs['ti'].xcom_pull(task_ids='dataPrep1', key='prep1_path')
    df = pd.read_csv(prep1_path)

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

    return df


def upload(**kwargs):
    
    SHEETS_ID = os.getenv('SHEETS_ID')
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(os.getenv('SHEETS_KEY')), ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID)

    prep2_path = kwargs['ti'].xcom_pull(task_ids='dataPrep2', key='prep2_path')
    df = pd.read_csv(prep2_path)
    
    prepared = sheet.worksheet("Prepared")
    prepared.clear()

    set_with_dataframe(prepared, df)

    logging.info("Data successfully saved to Google Sheets")


with DAG(
    'prepare',
    start_date=datetime(2023, 1, 1),
    schedule_interval='@daily',
) as dag:

    download_data = PythonOperator(
        task_id='download_data',
        python_callable=download,
    )

    dataPrep1_task = PythonOperator(
        task_id='dataPrep1',
        python_callable=dataPrep1,
        provide_context=True
    )
    dataPrep2_task = PythonOperator(
        task_id='dataPrep2',
        python_callable=dataPrep2,
        provide_context=True
    )

    save_data_to_sheets = PythonOperator(
        task_id='upload_data',
        python_callable=upload,
        provide_context=True
    )

    download_data >> dataPrep1_task >> dataPrep2_task >> save_data_to_sheets

