import lightgbm as lgb
import pandas as pd 
import numpy as np 
import joblib 
from preprocessing import load_and_preprocess 


MODEL_PATH = "models/lightgbm.txt"


def load_model():
    model = lgb.Booster(model_file = MODEL_PATH)
    return model 

def predict_batch(model, df: pd.DataFrame):
    preds = model.predict(df)
    return preds


def predict_single(model , instance_dict: dict):
    df = pd.DataFrame([instance_dict])
    preds = model.predict(df)
    return preds[0]


if __name__ == "__main__":
    model = load_model()
    df = load_and_preprocess(sample_frac=0.01)

    X = df.drop(columns=['label'])

    preds = predict_batch(model, X)
    print("Sample batch predictions:", preds[:10])

    sample_dict = X.iloc[0].to_dict()
    single_pred = predict_single(model, sample_dict)
    print("Single prediction:", single_pred)