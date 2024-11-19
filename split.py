import json
import logging
import gspread
import kagglehub
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from oauth2client.service_account import ServiceAccountCredentials

def download_and_split_data():


    path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")
    logging.info(f"Downloaded dataset to {path}")

    for file_name in os.listdir(path):
        if os.path.exists(f"./{file_name}"):
            os.remove(f"./{file_name}")
        shutil.move(os.path.join(path, file_name), "./")

    src = pd.read_csv("./loan_data.csv")
    data70, data30 = train_test_split(src, test_size=0.3, random_state=80)

    logging.info(f"Total records: {src.shape[0]}")
    logging.info(f"Modelowy (70%): {data70.shape[0]}")
    logging.info(f"Douczeniowy (30%): {data30.shape[0]}")

    SHEETS_ID = os.getenv('SHEETS_ID')

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(os.getenv('SHEETS_KEY')), ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID)
    
    modelowy = sheet.worksheet("Modelowy")
    modelowy.clear()
    modelowy.update([data70.columns.values.tolist()] + data70.values.tolist())

    douczeniowy = sheet.worksheet("Douczeniowy")
    douczeniowy.clear()
    douczeniowy.update([data30.columns.values.tolist()] + data30.values.tolist())

    logging.info("Data successfully saved to Google Sheets")

if __name__ == "__main__":
    download_and_split_data()
