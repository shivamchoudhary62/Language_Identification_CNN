import streamlit as st
import requests

st.title("🎙️ Language Identification System")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if st.button("Predict Language") and uploaded_file is not None:
    with st.spinner("Analyzing audio..."):
        # Send file to your FastAPI backend
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
        response = requests.post("http://127.0.0.1:8000/predict/", files=files)
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Language: **{result['predicted_language'].upper()}**")
            st.info(f"Confidence: {result['confidence']}")
        else:
            st.error("Error connecting to backend.")