
import logging
import pickle
from airflow import DAG
import gspread
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime
import logging
import json
from google.oauth2.service_account import Credentials

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
    credentials = Credentials.from_service_account_info(json.loads(os.getenv('SHEETS_KEY')), scopes=['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
      
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID).worksheet("Prepared")

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

def train():

    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv', nrows=25000)

    cat_cols = ['person_education', 'person_home_ownership', 'loan_intent']
    
    X = df.drop('loan_status', axis=1)
    X = pd.get_dummies(X, drop_first=True, columns=cat_cols)
    
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    

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
) as dag:

    download_task = PythonOperator(
        task_id='download_task',
        python_callable=download,
    )

    train_task = PythonOperator(
        task_id='train_task',
        python_callable=train,
    )


    download_task >> train_task 