import json
import logging
from airflow import DAG
import gspread
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from oauth2client.service_account import ServiceAccountCredentials
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
import pickle
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
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
    sheet = client.open_by_key(SHEETS_ID).worksheet("Prepared")

    sheetValues = sheet.get_all_values()

    df = pd.DataFrame(sheetValues)
    df.columns = df.iloc[0]
    df = df[1:]
    logging.info("downloaded")

    df.to_csv('/opt/airflow/processed_data/processed_data.csv', index=False)

def autoML():

    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')

    df = pd.get_dummies(df, drop_first=True)
    feats = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
       'credit_score',  'person_gender_male',
       'person_education_Bachelor', 'person_education_Doctorate',
       'person_education_High School', 'person_education_Master',
       'person_home_ownership_OTHER', 'person_home_ownership_OWN',
       'person_home_ownership_RENT', 'loan_intent_EDUCATION',
       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
       'loan_intent_PERSONAL', 'loan_intent_VENTURE',
       'previous_loan_defaults_on_file_Yes']
    
    target = 'loan_status'

    train, test = train_test_split(df, test_size=0.3)

    X_train = train.loc[:, feats]
    y_train = train[target]

    X_test = test.loc[:, feats] 
    y_test= test[target]

    tpot = TPOTClassifier(cv=5, verbosity=3, generations=3, population_size=50)

    tpot.fit(X_train, y_train)


    score = tpot.score(X_test, y_test)

    tpot.export("/opt/airflow/models/model.py")

    with open("/opt/airflow/reports/evaluation_report.txt", 'w') as report_file:
        report_file.write(f"Model Evaluation Report\n")
        report_file.write(f"------------------------\n")
        report_file.write(f"Accuracy: {score * 100:.2f}%\n")
    
with DAG(
    'train_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    download_task = PythonOperator(
        task_id='download_task',
        python_callable=download,
    )

    autoML_task = PythonOperator(
        task_id='autoML_task',
        python_callable=autoML,
    )


    download_task >> autoML_task 