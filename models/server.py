import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import io

app = FastAPI()

with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

@app.post("/")
async def pred(file: UploadFile = File(...)):
    if file.content_type == "text/csv":
        csv_bytes = await file.read()
        df = pd.read_csv(io.BytesIO(csv_bytes))
    elif file.content_type == "application/json":
        json_bytes = await file.read()
        df = pd.read_json(io.BytesIO(json_bytes))
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and JSON files are supported.")
    
    try:
        pred = model.predict(df)
        df['prediction'] = pred
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    result = df.to_dict(orient="records")
    return result