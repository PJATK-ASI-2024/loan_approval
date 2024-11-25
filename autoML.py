import json
import logging
import pickle
import gspread
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tpot import TPOTClassifier
from oauth2client.service_account import ServiceAccountCredentials


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('log.txt'),
        logging.StreamHandler()
    ]
)

def download_Modelowy_sheet():

    SHEETS_ID = os.getenv('SHEETS_ID')
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(os.getenv('SHEETS_KEY')), ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive'])
   
    client = gspread.authorize(credentials)
    sheet = client.open_by_key(SHEETS_ID).worksheet("Modelowy")

    sheetValues = sheet.get_all_values()

    df = pd.DataFrame(sheetValues)
    df.columns = df.iloc[0]
    df = df[1:] 
    return df


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
    df = download_Modelowy_sheet()
    X_train, y_train, X_test, y_test = dataPrep(df)
    # autoML(X_train, y_train)
    randomForest(X_train, y_train, X_test, y_test)
    XGB(X_train, y_train, X_test, y_test)



