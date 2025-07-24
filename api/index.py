from fastapi import FastAPI
import joblib   
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

target_names = load_iris().target_names
print("Target names:", target_names)
app = FastAPI()

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float 
    
md = joblib.load('./api/mlmodel.joblib')
@app.get("/prediction")
async def predict(data: IrisData):
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = md.predict(input_data)
    predicted_class = target_names[prediction[0]]
    return {"prediction": predicted_class}
