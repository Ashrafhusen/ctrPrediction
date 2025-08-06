from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


def frequency_encode(df, cols):
    for col in cols:
        freq = df[col].value_counts()
        df[col + "_freq"] = df[col].map(freq)
    return df


def target_encode(df, cols, target='label'):
    for col in cols:
        target_mean = df.groupby(col)[target].mean()
        df[col + "_target_enc"] = df[col].map(target_mean)
    return df


def cross_features(df, col_pairs):
    for col1, col2 in col_pairs:
        if col1 in df.columns and col2 in df.columns:
            new_col = f"{col1}_{col2}_cross"
            df[new_col] = df[col1].astype(str) + "_" + df[col2].astype(str)
            le = LabelEncoder()
            df[new_col] = le.fit_transform(df[new_col])
    return df


def log_transform(df, cols):
    for col in cols:
        df[col + "_log"] = np.log1p(df[col].astype(float))
    return df


def load_and_preprocess(sample_frac=0.1):
    dataset = load_dataset("reczoo/Criteo_x1", split="train").select(range(50000))






    df = dataset.to_pandas()

    int_features = [col for col in df.columns if col.startswith("I")]
    cat_features = [col for col in df.columns if col.startswith("C")]

    df[int_features] = df[int_features].fillna(-1)
    df[cat_features] = df[cat_features].fillna("-1")

    # Encode categorical features
    for col in cat_features:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    for col in int_features:
        df[col] = df[col].astype('float32')

    df['label'] = df['label'].astype("int")

    df = log_transform(df, int_features)
    df = frequency_encode(df, cat_features)
    df = target_encode(df, cat_features)

    cross_pairs = [("C1", "C2"), ("C3", "C5")]
    df = cross_features(df, cross_pairs)

    df["missing_count"] = (df[cat_features + int_features] == "-1").sum(axis=1)

    return df
