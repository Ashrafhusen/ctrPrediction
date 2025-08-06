import lightgbm as lgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib 
import mlflow 
import mlflow.lightgbm 
from preprocessing import load_and_preprocess 
import os   
import json
import pickle


def train_ctr_model():
    df = load_and_preprocess()

    X = df.drop(columns = ['label'])
    y = df['label']


    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size = 0.2, random_state = 42, stratify = y
    )


    train_data = lgb.Dataset(X_train, label = y_train)
    val_data = lgb.Dataset(X_val, label = y_val)


    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "verbosity": -1,
         "seed": 42
    }


    with mlflow.start_run():
        model = lgb.train(
            params,
            train_data,
            valid_sets = [train_data, val_data],
            valid_names = ['train', 'val'],
            num_boost_round = 500,
        )

        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        print(f"AUC on validation set: {auc:.4f}")


        mlflow.log_params(params)
        mlflow.log_metric('auc', auc)
        mlflow.lightgbm.log_model(model, "model")

        os.makedirs("models", exist_ok = True)
        model.save_model("models/lightgbm.txt")
        joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
        


if __name__ == "__main__":
    train_ctr_model()
