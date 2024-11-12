import logging
import kagglehub
import os
import shutil
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from xgboost import XGBClassifier
import ydata_profiling as yp
from tpot import TPOTClassifier

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

    profile = yp.ProfileReport(df)
    profile.to_file("profile_report.html")

def dataPrep(df):

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

    scaler = StandardScaler()

    train[feats] = scaler.fit_transform(train[feats])
    test[feats] = scaler.fit_transform(test[feats]) 

    X_train = train.loc[:, feats]
    y_train = train[target]

    X_test = test.loc[:, feats] 
    y_test= test[target]

    return X_train, y_train, X_test, y_test

def autoML(X_train, y_train):

    tpot = TPOTClassifier(cv=5, verbosity=3, generations=3, population_size=50)

    tpot.fit(X_train, y_train)

    results = pd.DataFrame(tpot.evaluated_individuals_).T

    results.drop(columns=['predecessor'])

    results = results.sort_values(by='internal_cv_score', ascending=False)

    print(results)

def randomForest(X_train, y_train, X_test, y_test): 
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Random forest model accuracy: {accuracy * 100:.2f}%")
    print(f"Random forest model mae: {mae:.2f}%")

def XGB(X_train, y_train, X_test, y_test): 
    
    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"XGB model accuracy: {accuracy * 100:.2f}%")
    print(f"XGB model mae: {mae:.2f}%")

if __name__ == "__main__":
    df = download_and_split_data()
    with PdfPages('plots.pdf') as pdf_pages:
        EDA(df, pdf_pages)
    df = pd.read_csv("loan_data_70.csv")
    X_train, y_train, X_test, y_test = dataPrep(df)
    # autoML(X_train, y_train)
    randomForest(X_train, y_train, X_test, y_test)
    XGB(X_train, y_train, X_test, y_test)
