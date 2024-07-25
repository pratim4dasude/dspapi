from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from typing import Dict

model = joblib.load('api/model.pkl')

class InputData(BaseModel):
    Category: str
    Accident_type: str
    Year: int
    Month: int


category_encoding = {
    'Verkehrsunfälle': 0,
    'Alkoholunfälle': 1,
    'Fluchtunfälle': 2
}

accident_type_encoding = {
    'insgesamt': 0,
    'Verletzte und Getötete': 1,
    'mit Personenschäden': 2
}

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(input_data: InputData):

    category_encoded = category_encoding.get(input_data.Category, -1)
    accident_type_encoded = accident_type_encoding.get(input_data.Accident_type, -1)

    if category_encoded == -1:
        raise HTTPException(status_code=400, detail="Invalid Category")
    if accident_type_encoded == -1:
        raise HTTPException(status_code=400, detail="Invalid Accident Type")


    data = [[category_encoded, accident_type_encoded, input_data.Year, input_data.Month]]


    try:
        prediction = model.predict(data)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    return {"prediction": prediction}
