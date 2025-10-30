#!/usr/bin/env python3
import json
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Deployment Assignment")


@app.post("/predict")
async def post_prediction(req: Request):

    data = await req.json()
    print(data)
    response = {"Prediction": float(await predict(data))}
    return JSONResponse(content=json.dumps(response), status_code=200)


async def predict(customer):
    dv, model = None, None
    with open("pipeline_v1.bin", "rb") as file:
        dv, model = pickle.load(file)

    X = dv.transform(customer)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


if __name__ == "__main__":
    customer = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0,
    }
    value = predict(customer)
    print(f"Prediction: {value}")
