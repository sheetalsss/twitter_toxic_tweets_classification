import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("/Users/sheetals/Downloads/twitter_toxic_tweets_classification/toxic_svm_pipeline.joblib")
# model = bundle["model"]
# threshold = bundle["threshold"]

st.title("Toxic Tweet Detector")

text = st.text_area("Enter text")

if st.button("Predict"):
    score = model.decision_function([text])[0]
    pred = int(score)


    st.write(f"Score: {score:.3f}")

    if pred == 1:
        st.error("Toxic")
    else:
        st.success("Non-toxic")
