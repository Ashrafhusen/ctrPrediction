## Real-Time CTR Prediction Pipeline (Criteo Private Ad + LightGBM)

An end-to-end Click-Through Rate (CTR) prediction pipeline using the Criteo Private Ad dataset, featuring LightGBM, MLflow experiment tracking, and real-time inference via FastAPI or Streamlit. The project demonstrates a production-grade ML workflow, including data preprocessing, feature engineering, model training, evaluation, and real-time predictions.

## Tech Stack

Machine Learning: LightGBM, scikit-learn, Pandas, NumPy

Experiment Tracking & Model Management: MLflow

Inference / UI: FastAPI (API) & Streamlit (interactive frontend)

Optional Orchestration: Airflow / Prefect


## Key Steps:

Data Preprocessing: Handle missing values, type conversions, log transforms.

Feature Engineering: Frequency encoding, target encoding, cross features, and log transformation.

Model Training: LightGBM with AUC and logloss evaluation, train/validation split.

Experiment Tracking: Log parameters, metrics, and artifacts with MLflow.

Real-Time Prediction:

FastAPI endpoints for single and batch predictions

Streamlit app for interactive frontend

## Dataset Description

Criteo Private Ad Dataset (privacy-safe version of Criteo 1TB):

Integer features (I1–I13): User/product numeric context features

Categorical features (C1–C26): Hashed user/product/ad metadata

Target: click (0 or 1)

