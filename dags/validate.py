
import logging
import pickle
from airflow import DAG
import gspread
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from airflow.operators.email import EmailOperator
import json
from google.oauth2.service_account import Credentials
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from airflow.models import Variable



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
    sheet = client.open_by_key(SHEETS_ID).worksheet("Douczeniowy")

    sheetValues = sheet.get_all_values()

    df = pd.DataFrame(sheetValues)
    df.columns = df.iloc[0]
    df = df[1:]

    df['person_gender'] = df['person_gender'].apply(lambda x: 1 if x == 'male' else 0)
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].apply(lambda x: 1 if x == 'Yes' else 0)

    num_cols = ['person_age', 'person_income',
        'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score']
    for col in num_cols:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)



    logging.info("downloaded")

    directory = '/opt/airflow/processed_data'
    file_path = os.path.join(directory, 'douczeniowy.csv')

    os.makedirs(directory, exist_ok=True)

    df.to_csv(file_path, index=False)

def validate():
        
    df = pd.read_csv('/opt/airflow/processed_data/douczeniowy.csv')



    with open('/opt/airflow/models/model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    cat_cols = ['person_education', 'person_home_ownership', 'loan_intent']
    
    X = df.drop('loan_status', axis=1)
    X = pd.get_dummies(X, drop_first=True, columns=cat_cols)
    
    y = df['loan_status']

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    mail_content=f"""
        Model validation report:</br>
        Accuracy: {accuracy * 100:.2f}%</br>
        MAE: {mae:.2f}%")
    """

    Variable.set("mail_content", mail_content)

    if(accuracy < 0.99 or mae > 0.01):
        raise AirflowException('Model evaluation failed')

with DAG(
    'validate_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
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
        html_content="""<h1>Random forest model validation or tests failed.</h1>
            <p>{{ var.value.mail_content }}</p>""",
        trigger_rule='one_failed'
    )

    download_task >> validate_task >> email_task