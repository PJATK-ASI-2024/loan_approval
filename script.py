import kagglehub
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

script_folder = os.path.dirname(os.path.realpath(__file__))

downloaded_path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")

for file_name in os.listdir(downloaded_path):
    if os.path.exists(f"./{file_name}"):
        os.remove(f"./{file_name}")
    shutil.move(os.path.join(downloaded_path, file_name), "./")

src = pd.read_csv("./loan_data.csv")
data70, data30 = train_test_split(src, test_size=0.3)

print(src.shape[0])
print(data70.shape[0])
print(data30.shape[0])

data70.to_csv(os.path.join("./", "loan_data_70.csv"), index=False)
data30.to_csv(os.path.join("./", "loan_data_30.csv"), index=False)

df = pd.read_csv("./loan_data_70.csv")
print(df.shape[0])