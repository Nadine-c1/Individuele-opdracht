import streamlit as st
import requests
from PIL import Image
import io
import base64
import cv2

st.set_page_config(page_title="Anomaly Detector", layout="centered")
st.title("🧠 Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("📷 Upload een afbeelding", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Ingevoerde afbeelding", use_container_width=True)

    if st.button("Detecteer"):
        with st.spinner("Analyse bezig..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
            )

            if response.status_code == 200:
                result = response.json()
                
                if "error" in result:
                    st.error(f"❌ API-fout: {result['error']}")
                else:
                    st.metric("🔍 MSE", f"{result['mse']:.5f}")
                    if result["is_anomaly"]:
                        st.error("🚨 Afwijking gedetecteerd!")
                    else:
                        st.success("✅ Normale kaart")

                    # Reconstructie tonen
                    recon_data = base64.b64decode(result["reconstruction"])
                    recon_img = Image.open(io.BytesIO(recon_data))
                    st.image(recon_img, caption="🔁 Gereconstrueerde afbeelding", use_container_width=True)

            else:
                st.error("❌ Fout bij versturen naar API.")