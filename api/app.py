from fastapi import FastAPI 
from pydantic import BaseModel
from typing import List 
import lightgbm as lgb 
import pandas as pd 
import joblib

from src.preprocessing import (
    log_transform, frequency_encode, target_encode, cross_features
)

app = FastAPI(title="Predict CTR")

MODEL_PATH = "models/lightgbm.txt"
FEATURE_PATH = "models/feature_names.pkl"

model = lgb.Booster(model_file=MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

class CTRRequest(BaseModel):
    features: dict

class BatchCTRRequest(BaseModel):
    instances: List[CTRRequest]

@app.get("/")
def read_root():
    return {'status': "API is Now Live"}

@app.post("/predict")
def predict_ctr(req: CTRRequest):
    df = pd.DataFrame([req.features])

    int_features = [col for col in df.columns if col.startswith("I")]
    cat_features = [col for col in df.columns if col.startswith("C")]

    df[int_features] = df[int_features].fillna(-1).astype("float32")
    df[cat_features] = df[cat_features].fillna("-1").astype(str)

    df = log_transform(df, int_features)
    df = frequency_encode(df, cat_features)
    df["dummy"] = 0
    df = target_encode(df, cat_features, target="dummy")
    df.drop(columns=["dummy"], inplace=True)

    df = cross_features(df, [("C1", "C2"), ("C3", "C5")])
    df["missing_count"] = (df[cat_features + int_features] == "-1").sum(axis=1)

    df.drop(columns=cat_features, inplace=True)

    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    pred = model.predict(df)[0]
    return {"click_probability": round(pred, 4)}

@app.post("/predict_batch")
def predict_batch(req: BatchCTRRequest):
    records = [r.features for r in req.instances]
    df = pd.DataFrame(records)

    int_features = [col for col in df.columns if col.startswith("I")]
    cat_features = [col for col in df.columns if col.startswith("C")]

    df[int_features] = df[int_features].fillna(-1).astype("float32")
    df[cat_features] = df[cat_features].fillna("-1").astype(str)

    df = log_transform(df, int_features)
    df = frequency_encode(df, cat_features)
    df["dummy"] = 0
    df = target_encode(df, cat_features, target="dummy")
    df.drop(columns=["dummy"], inplace=True)

    df = cross_features(df, [("C1", "C2"), ("C3", "C5")])
    df["missing_count"] = (df[cat_features + int_features] == "-1").sum(axis=1)

    df.drop(columns=cat_features, inplace=True)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    preds = model.predict(df)
    return {"predictions": [round(p, 4) for p in preds]}
