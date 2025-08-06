import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="CTR Predictor", page_icon="🎯", layout="centered")

st.title("Real-Time CTR Prediction")
st.markdown("Predict the click-through rate (CTR) for ad impressions.")

def make_single_prediction(features):
    try:
        response = requests.post("http://localhost:8001/predict", json={"features": features})
        if response.status_code == 200:
            result = response.json()
            prob = result["click_probability"]
            return prob, None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"API not reachable: {e}"

def make_batch_prediction(instances):
    try:
        response = requests.post("http://localhost:8001/predict_batch", json={"instances": instances})
        if response.status_code == 200:
            result = response.json()
            return result["predictions"], None
        else:
            return None, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"API not reachable: {e}"

with st.form("single_predict_form"):
    st.subheader("Input Single Ad Features")

    i1 = st.number_input("I1 (integer)", min_value=0, max_value=1000, value=5)
    i2 = st.number_input("I2 (integer)", min_value=0, max_value=1000, value=0)
    i3 = st.number_input("I3 (integer)", min_value=0, max_value=1000, value=10)

    c1 = st.text_input("C1 (category)", value="a")
    c2 = st.text_input("C2 (category)", value="b")
    c3 = st.text_input("C3 (category)", value="c")
    c5 = st.text_input("C5 (category)", value="e")

    submitted_single = st.form_submit_button("🔮 Predict Single")

    if submitted_single:
        features = {
            "I1": i1,
            "I2": i2,
            "I3": i3,
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C5": c5
        }
        prob, error = make_single_prediction(features)
        if error:
            st.error(f"{error}")
        else:
            st.success(f"Click Probability: **{round(prob * 100, 2)}%**")
            st.progress(min(1.0, prob))

st.markdown("---")


st.subheader("Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file with features (columns: I1,I2,I3,C1,C2,C3,C5)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = {"I1", "I2", "I3", "C1", "C2", "C3", "C5"}
        if not required_cols.issubset(set(df.columns)):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            st.write(f"Preview of uploaded data ({df.shape[0]} rows):")
            st.dataframe(df.head())

            if st.button("🔮 Predict Batch"):
                instances = df.to_dict(orient="records")
                preds, error = make_batch_prediction(instances)
                if error:
                    st.error(f"❌ {error}")
                else:
                    df["click_probability"] = preds
                    df["click_probability_percent"] = df["click_probability"].apply(lambda x: round(x * 100, 2))
                    st.success(f"Batch Predictions completed for {len(preds)} rows.")
                    st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
