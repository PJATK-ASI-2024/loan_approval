import kagglehub
import os
import shutil

script_folder = os.path.dirname(os.path.realpath(__file__))

downloaded_path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")

for file_name in os.listdir(downloaded_path):
    shutil.move(os.path.join(downloaded_path, file_name), "./")

from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("./loan_data.csv")
data70, data30 = train_test_split(data, test_size=0.3)

print(data.shape[0])
print(data70.shape[0])
print(data30.shape[0])

data70.to_csv(os.path.join("./", "loan_data.csv"), index=False)
data30.to_csv(os.path.join("./", "loan_data_30.csv"), index=False)

data2 = pd.read_csv("./loan_data.csv")
print(data2.shape[0])