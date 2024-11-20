
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


def download_Modelowy_sheet():

    SHEETS_ID = os.getenv('SHEETS_ID')
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(os.getenv('SHEETS_KEY')), ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID).worksheet("Modelowy")

    sheetValues = sheet.get_all_values()

    df = pd.DataFrame(sheetValues)

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

    profile = yp.ProfileReport(df)
    profile.to_file("profile_report.html")

if __name__ == "__main__":
    df = download_Modelowy_sheet()
    with PdfPages('plots.pdf') as pdf_pages:
        EDA(df, pdf_pages)