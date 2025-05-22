from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import traceback
import base64
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Model loading
try:
    model = tf.keras.models.load_model("../model/autoencoder.h5")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# ✅ Preprocess helper
# 👇 Preprocess helper
def preprocess(image: Image.Image):
    try:
        img = image.resize((224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.grayscale_to_rgb(img)
        img = img / 255.0
        return np.expand_dims(img, axis=0)

    except Exception as e:
        print("❌ Preprocessing error:", e)
        raise

# ✅ Predict route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        input_img = preprocess(image)

        if model is None:
            return {"error": "Model not loaded"}

        reconstructed = model.predict(input_img)
        mse = np.mean((input_img - reconstructed) ** 2)
        is_anomaly = mse > 0.26  # Pas eventueel je threshold aan

        # Maak reconstructie PNG + base64
        recon_img = (reconstructed[0] * 255).astype("uint8")
        _, buffer = cv2.imencode(".png", recon_img)
        recon_bytes = base64.b64encode(buffer).decode("utf-8")

        return {
            "mse": float(mse),
            "is_anomaly": bool(is_anomaly),
            "reconstruction": recon_bytes
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}