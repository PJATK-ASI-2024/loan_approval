import logging
import kagglehub
import os
import shutil
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

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

def EDA(df, pdf_pages):

    print(df)
    print(df.info())

    print(df.describe().T)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    print(df[cat_cols].describe().T)
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

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

if __name__ == "__main__":
    with PdfPages('plots.pdf') as pdf_pages:
        df = download_and_split_data()
        EDA(df, pdf_pages)