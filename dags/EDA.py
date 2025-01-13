from datetime import datetime
from airflow import DAG
import os
import os
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import ydata_profiling as yp

import pandas as pd
from airflow.operators.python import PythonOperator

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

with DAG(
    'EDA',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
     EDA_task = PythonOperator(
        task_id='EDA_task',
        python_callable=EDA,
    )