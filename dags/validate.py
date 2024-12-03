
import logging
import pickle
from airflow import DAG
import gspread
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from airflow.operators.email_operator import EmailOperator
import json
from oauth2client.service_account import ServiceAccountCredentials
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
    sheet = client.open_by_key(SHEETS_ID).worksheet("Douczeniowy")

    sheetValues = sheet.get_all_values()

    df = pd.DataFrame(sheetValues)
    df.columns = df.iloc[0]
    df = df[1:]
    logging.info("downloaded")

    directory = '/opt/airflow/processed_data'
    file_path = os.path.join(directory, 'douczeniowy.csv')

    os.makedirs(directory, exist_ok=True)

    df.to_csv(file_path, index=False)

def validate():

    df = pd.read_csv('/opt/airflow/processed_data/douczeniowy.csv', nrows=10000)
    with open('/opt/airflow/models/model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    X = df.drop('loan_status', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    y = df['loan_status']

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    if(accuracy < 0.9 | mae < 0.9):


    

    directory = '/opt/airflow/models'
    file_path = os.path.join(directory, 'model.py')

    os.makedirs(directory, exist_ok=True)

    with open('/opt/airflow/models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    directory = '/opt/airflow/reports'
    file_path = os.path.join(directory, 'evaluation_report.txt')

    os.makedirs(directory, exist_ok=True)

    with open(file_path, 'w') as report_file:
        report_file.write(f"Model Evaluation Report\n")
        report_file.write(f"------------------------\n")
        report_file.write(f"Random forest model accuracy: {accuracy * 100:.2f}%")
        report_file.write(f"Random forest model mae: {mae:.2f}%")
    
with DAG(
    'train_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    email_on_failure=True
) as dag:

    download_task = PythonOperator(
        task_id='download_task',
        python_callable=download,
    )

    validate_task = PythonOperator(
        task_id='validate_task',
        python_callable=validate,
    )

    email_task = EmailOperator(
        task_id='email_task',
        to='s25361@pjwstk.edu.pl',
        subject='Model validation failed',
        html_content='<p>This is a test email sent by Apache Airflow.</p>',
    )

    download_task >> validate_task 