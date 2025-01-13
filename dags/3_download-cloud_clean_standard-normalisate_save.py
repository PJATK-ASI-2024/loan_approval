from datetime import datetime
import json
import logging
from airflow import DAG
import gspread
import os
import json
import gspread
import os
import pandas as pd
from google.oauth2.service_account import Credentials
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def download_cloud():
    logging.info("downloading...")

    SHEETS_ID = os.getenv('SHEETS_ID')
    credentials = Credentials.from_service_account_info(json.loads(os.getenv('SHEETS_KEY')), scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID).worksheet("Modelowy")

    sheetValues = sheet.get_all_values()

    df = pd.DataFrame(sheetValues)
    df.columns = df.iloc[0]
    df = df[1:]

    num_cols = ['person_age', 'person_income',
    'person_emp_exp', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'credit_score']
    for col in num_cols:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)


    logging.info("downloaded")


    directory = '/opt/airflow/processed_data'
    file_path = os.path.join(directory, 'processed_data.csv')

    os.makedirs(directory, exist_ok=True)

    df.to_csv(file_path, index=False)


def clean():

    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')

    logging.info("preparing part 1...")

    df['person_gender'] = df['person_gender'].apply(lambda x: 1 if x == 'male' else 0)
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].apply(lambda x: 1 if x == 'Yes' else 0)



    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        logging.info(f"Found {num_duplicates} duplicates. Removing them.")
        df = df.drop_duplicates()
    df.replace('', np.nan, inplace=True)

    logging.info("Finished prep part 1...")
    logging.info("saving file...")

    df.to_csv('/opt/airflow/processed_data/processed_data.csv', index=False)
    logging.info("saved file")

def standard_normalisate():

    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')

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

    df.to_csv('/opt/airflow/processed_data/processed_data.csv', index=False)



def save():
    
    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')

    SHEETS_ID = os.getenv('SHEETS_ID')
    credentials = Credentials.from_service_account_info(json.loads(os.getenv('SHEETS_KEY')), scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID)
    
    prepared = sheet.worksheet("Prepared")
    prepared.clear()

    set_with_dataframe(prepared, df)

    logging.info("Data successfully saved to Google Sheets")


with DAG(
    '3_download-cloud_clean_standard-normalisate_save',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    download_cloud_task = PythonOperator(
        task_id='download_cloud_task',
        python_callable=download_cloud,
    )

    clean_task = PythonOperator(
        task_id='clean_task',
        python_callable=clean,
    )

    standard_normalisate_task = PythonOperator(
        task_id='standard_normalisate_task',
        python_callable=standard_normalisate,
    )

    save_task = PythonOperator(
        task_id='save_task',
        python_callable=save,
    )


    download_cloud_task >> clean_task >> standard_normalisate_task >> save_task
