import streamlit as st
import requests
from PIL import Image
import io
import base64
import os

st.set_page_config(page_title="Anomaly Detector", layout="centered")
st.title("🧠 Anomaly Detection Dashboard")

# 📂 Testfolder pad
TEST_FOLDER = "/Users/nadine/Documents/Engineering/Jaar 3/Data science /Blok 2/Smart industry /Individuele opdracht/test"

# 🔎 Alle afbeeldingspaden verzamelen
image_paths = []
for root, _, files in os.walk(TEST_FOLDER):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            image_paths.append(os.path.join(root, file))

# 🔽 Selectbox met relatieve paden
file_names = [os.path.relpath(path, TEST_FOLDER) for path in image_paths]
selected_file = st.selectbox("📁 Kies een testafbeelding", file_names)

if selected_file:
    full_path = os.path.join(TEST_FOLDER, selected_file)

    with open(full_path, "rb") as f:
        file_bytes = f.read()
        image = Image.open(io.BytesIO(file_bytes))
        st.image(image, caption="🖼️ Ingevoerde afbeelding", use_container_width=True)

    if st.button("Detecteer"):
        with st.spinner("🔍 Analyse bezig..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={"file": (selected_file, file_bytes, "image/jpeg")}
            )

            if response.status_code == 200:
                result = response.json()

                if "error" in result:
                    st.error(f"❌ API-fout: {result['error']}")
                else:
                    st.metric("📏 MSE", f"{result['mse']:.5f}")
                    if result["is_anomaly"]:
                        st.error("🚨 Afwijking gedetecteerd!")
                    else:
                        st.success("✅ Normale kaart")

                    # 🔁 Reconstructie tonen
                    recon_data = base64.b64decode(result["reconstruction"])
                    recon_img = Image.open(io.BytesIO(recon_data))
                    st.image(recon_img, caption="🔁 Gereconstrueerde afbeelding", use_container_width=True)

            else:
                st.error("❌ Fout bij versturen naar API.")