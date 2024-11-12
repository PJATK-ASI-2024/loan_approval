import logging
import kagglehub
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)

def download_and_split_data():

    path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")
    logging.info(f"Downloaded dataset to {path}")

    for file_name in os.listdir(path):
        if os.path.exists(f"./{file_name}"):
            os.remove(f"./{file_name}")
        shutil.move(os.path.join(path, file_name), "./")

    src = pd.read_csv("./loan_data.csv")
    data70, data30 = train_test_split(src, test_size=0.3)

    logging.info(f"Total records: {src.shape[0]}")
    logging.info(f"Training set size (70%): {data70.shape[0]}")
    logging.info(f"Test set size (30%): {data30.shape[0]}")

    data70.to_csv(os.path.join("./", "loan_data_70.csv"), index=False)
    data30.to_csv(os.path.join("./", "loan_data_30.csv"), index=False)

    df = pd.read_csv("./loan_data_70.csv")
    logging.info(f"Loaded training set size from file: {df.shape[0]}")

    return df

if __name__ == "__main__":
    df = download_and_split_data()