import logging
import pickle
from airflow import DAG
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from airflow.operators.email import EmailOperator
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

def validate():
        
    df = pd.read_csv('/opt/airflow/processed_data/processed_data.csv')
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

    if(accuracy < 0.9 or mae > 0.1):
        raise AirflowException('Model evaluation failed')

with DAG(
    '5_monitoring',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

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

    validate_task >> email_task