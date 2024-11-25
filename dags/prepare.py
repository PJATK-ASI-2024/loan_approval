from datetime import datetime
import json
import logging
from airflow import DAG
import gspread
import os
import json
import gspread
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import ydata_profiling as yp
from oauth2client.service_account import ServiceAccountCredentials
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


    directory = '/opt/airflow/processed_data'
    file_path = os.path.join(directory, 'processed_data.csv')

    os.makedirs(directory, exist_ok=True)

    df.to_csv(file_path, index=False)


def dataPrep1():

    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')

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

    df.to_csv('/opt/airflow/processed_data/processed_data.csv', index=False)
    logging.info("saved file")

def dataPrep2():

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

def EDA():

    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')

    print(df)
    print(df.info())

    print(df.describe().T)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    print(df[cat_cols].describe().T)
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    directory = '/opt/airflow/visualizations'
    file_path = os.path.join(directory, 'plots.pdf')

    os.makedirs(directory, exist_ok=True)

    with PdfPages(file_path) as pdf_pages:

        for col in num_cols:

            plt.figure(figsize=(16, 9))

            plt.subplot(1, 2, 1)
            plt.hist(df[col].dropna())
            plt.title(f'{col} Distribution')
            plt.xlabel(col)
            plt.ylabel('Count')

            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f'{col} Boxplot')
            plt.xlabel(col)

            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x)))

            pdf_pages.savefig()
            plt.close()
        
        for col in cat_cols:
            
            plt.figure(figsize=(16, 9))

            plt.subplot(1, 2, 1)
            sns.countplot(x=df[col])
            plt.title(f'{col} Distribution')
            plt.xlabel(col)
            plt.ylabel('Count')

            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f'{col} Boxplot')
            plt.xlabel(col)

            plt.xticks(rotation=45)

            pdf_pages.savefig()
            plt.close()

        plt.figure(figsize=(12, 7))
        sns.heatmap(df.drop(cat_cols, axis=1).corr(), annot = True, vmin = -1, vmax = 1)
        plt.xticks(rotation=45)
        pdf_pages.savefig()
        plt.close()

    print(df.isnull().sum())

    profile = yp.ProfileReport(df)

    file_path = os.path.join(directory, 'profile_report.html')

    profile.to_file(file_path)

    

def upload():
    
    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')

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

    EDA_task = PythonOperator(
        task_id='EDA_task',
        python_callable=EDA,
    )

    upload_task = PythonOperator(
        task_id='upload_task',
        python_callable=upload,
    )

    download_task >> dataPrep1_task >> dataPrep2_task >> upload_task
    download_task >> EDA_task
