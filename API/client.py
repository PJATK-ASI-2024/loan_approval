import os
import requests
import json

url = "http://localhost:5000"

def upload_file(file_path: str):

    file_extension = os.path.splitext(file_path)[1].lower()

    with open(file_path, "r") as file:
        file_content = file.read()

        if file_extension == ".csv":
            file_type = "text/csv"
        else:
            file_type = "application/json"

        files = {"file": (file_path, file_content, file_type)}

        response = requests.post(url, files=files)

        if response.status_code == 200:
            return(response.json())
        else:
            print("Error:", response.text)
            return None

if __name__ == "__main__":
    sample_data_csv = "./API/sample_data/sample_data.csv" 
    sample_data_json = "./API/sample_data/sample_data.json" 

    csv_res = upload_file(sample_data_csv)

    with open("./API/sample_data/sample_data_res/sample_data_csv_res.json", "w") as json_file:
        json.dump(csv_res, json_file, indent=4) 

    json_res = upload_file(sample_data_json)

    with open("./API/sample_data/sample_data_res/sample_data_json_res.json", "w") as json_file:
        json.dump(json_res, json_file, indent=4) 