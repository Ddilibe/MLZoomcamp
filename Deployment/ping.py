#!/usr/bin/env python3
import pickle
from flask import Flask, request, jsonify


app = Flask("Churn")
dv, model = None, None
output_file = "model_C=1.0.bin"

with open(output_file, "rb") as f_in:
    (dv, model) = pickle.load(f_in)


@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {"churn_prediction": y_pred, "churn": bool(churn)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=9696, host="0.0.0.0")
