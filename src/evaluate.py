import lightgbm as lgb
import pandas as pd  
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss 
from preprocessing import load_and_preprocess
import mlflow
import json
import os


MODEL_PATH  = 'models/lightgbm.txt'
METRICS_OUTPUT = "metrics/evaluation_metrics.json"

def evaluate():
    model = lgb.Booster(model_file = MODEL_PATH)
    df = load_and_preprocess(sample_frac=0.01)


    X  = df.drop(columns=["label"])
    y_true = df['label']

    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_pred_proba)
        }
    print("\n Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


    os.makedirs("metrics", exist_ok=True)
    with open(METRICS_OUTPUT, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n Metrics saved to {METRICS_OUTPUT}")

    with mlflow.start_run(run_name="evaluate_model"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(METRICS_OUTPUT)
        mlflow.log_param("eval_sample_size", len(df))




if __name__ == "__main__":
    evaluate()
