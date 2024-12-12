from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import io
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

app = FastAPI()

X, y = make_regression(n_features=5)
model = LinearRegression()
model.fit(X, y)

@app.post("/")
async def pred(file: UploadFile = File(...)):
    if file.content_type == "text/csv":
        csv_bytes = await file.read()
        df = pd.read_csv(io.BytesIO(csv_bytes))
        pred = model.predict(df)
        df['prediction'] = pred 
        return df
    elif file.content_type == "application/json":
        json_bytes = await file.read()
        df = pd.read_json(io.BytesIO(json_bytes))
        pred = model.predict(df)
        df['prediction'] = pred 
        return df
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and JSON files are supported.")