from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import uvicorn
import json
import io
from PIL import Image
import os
import gdown

app = FastAPI()

# ==========================================
# 1. Load model dan label
# ==========================================
MODEL_PATH = "model_tanaman_finetuned.keras"
LABEL_PATH = "class_labels.json"
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1SNgz17MqeFyTvWWxJgQ5ylyIz2TAtAZx/view?usp=sharing"
    gdown.download(url, MODEL_PATH, quiet=False)
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, 'r') as f:
    label_map = json.load(f)

# pastikan label dalam urutan index
class_labels = [label_map[str(i)] if str(i) in label_map else label_map[i] 
                for i in sorted(label_map.keys(), key=lambda x: int(x))]

# ==========================================
# 2. Endpoint prediksi
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
        label = class_labels[idx]

        return JSONResponse({
            "label": label,
            "probability": prob,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ==========================================
# 3. Jalankan server
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


