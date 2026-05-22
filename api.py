from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

app = FastAPI()

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    "behavioral_purchase_model.pkl"
)

model_features = joblib.load(
    "model_features.pkl"
)

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():

    return {
        "message":
        "Behavioral AI API Running"
    }

# =========================
# PREDICTION ROUTE
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    df = pd.read_csv(
        BytesIO(contents)
    )

    # =========================
    # FEATURE ENGINEERING
    # =========================

    df['engagement_score'] = (
        df['Administrative_Duration'] +
        df['Informational_Duration'] +
        df['ProductRelated_Duration']
    )

    df['product_focus_ratio'] = (
        df['ProductRelated'] /
        (df['Administrative'] + 1)
    )

    df['quick_decision_score'] = (
        df['PageValues'] /
        (df['ProductRelated_Duration'] + 1)
    )

    df['browsing_efficiency'] = (
        df['PageValues'] /
        (df['BounceRates'] + 0.001)
    )

    df['exit_intent_score'] = (
        df['ExitRates'] *
        df['PageValues']
    )

    df['informational_neglect'] = (
        df['ProductRelated'] -
        df['Informational']
    )

    # =========================
    # KEEP MODEL FEATURES
    # =========================

    df = df[model_features]

    # =========================
    # NUMERIC CONVERSION
    # =========================

    for col in model_features:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    df = df.fillna(0)

    # =========================
    # PREDICTIONS
    # =========================

    predictions = model.predict(df)

    probabilities = (
        model.predict_proba(df)[:, 1]
    )

    results = []

    for pred, prob in zip(
        predictions,
        probabilities
    ):

        results.append({

            "prediction":
            int(pred),

            "purchase_probability":
            round(float(prob * 100), 2)

        })

    return {
        "results": results
    }