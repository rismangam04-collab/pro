from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import uvicorn
import json
import io
import os
from PIL import Image
import gdown

app = FastAPI(title="Tanaman Disease Detector API")

# ==========================================
# 1. Load model dan label
# ==========================================
MODEL_PATH = "model_tanaman_finetuned.keras"
LABEL_PATH = "class_labels.json"
MODEL_URL = "https://drive.google.com/uc?id=1SNgz17MqeFyTvWWxJgQ5ylyIz2TAtAZx"

# Download model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {e}")

# Load label jika tersedia
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r") as f:
        label_map = json.load(f)
    class_labels = [label_map[str(i)] if str(i) in label_map else label_map[i]
                    for i in sorted(label_map.keys(), key=lambda x: int(x))]
else:
    print("Peringatan: file class_labels.json tidak ditemukan. Menggunakan indeks numerik.")
    class_labels = [str(i) for i in range(model.output_shape[-1])]

# ==========================================
# 2. Endpoint dasar
# ==========================================
@app.get("/")
async def root():
    return {"message": "API siap. Gunakan endpoint /predict untuk prediksi gambar."}

# ==========================================
# 3. Endpoint prediksi
# ==========================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((192, 192))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        pred = model.predict(img_array)
        idx = int(np.argmax(pred))
        prob = float(np.max(pred))
        label = class_labels[idx] if idx < len(class_labels) else str(idx)

        return JSONResponse({
            "label": label,
            "probability": prob
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==========================================
# 4. Run lokal
# ==========================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
