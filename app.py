from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import io
import cv2
import numpy as np
from ultralytics import YOLO
import os
import uvicorn
from typing import List, Dict

app = FastAPI(title="Saba Banana Counter API")

# CORS for frontend (same-origin + localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (best.pt for banana detection)
model_path = "./best.pt"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found: {model_path}")

model = YOLO(model_path)
print(f"Loaded model: {model_path}")

@app.get("/health")
async def health():
    return {"status": "ok", "model": model_path}

@app.post("/detect")
async def detect(image_b64: str):
    try:
        # Decode base64 to image
        image_data = base64.b64decode(image_b64.split(',')[1])  # Remove data:image/...;
        pil_image = Image.open(BytesIO(image_data)).convert('RGB')
        img_array = np.array(pil_image)

        # Run inference
        results = model(img_array, verbose=False)[0]

        # Filter banana detections (assume class 0 = banana)
        boxes = results.boxes
        if boxes is not None:
            confs = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            banana_indices = np.where(cls == 0)[0]  # class 0 = banana
            count = len(banana_indices)
            avg_confidence = np.mean(confs[banana_indices]) if count > 0 else 0.0
            detections = []
            if count > 0:
                for idx in banana_indices:
                    x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
                    detections.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": float(confs[idx])})
        else:
            count = 0
            avg_confidence = 0.0
            detections = []

        # Estimated weight: count * 2.5 kg
        estimated_weight = count * 2.5

        # Annotate image
        annotated_pil = pil_image.copy()
        draw = ImageDraw.Draw(annotated_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()

        if count > 0:
            for det in detections:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                conf = det['conf']
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                draw.text((x1, y1-25), f"Banana: {conf:.2f}", fill="lime", font=font)

        # Encode annotated image to base64
        buffered = BytesIO()
        annotated_pil.save(buffered, format="JPEG", quality=90)
        output_b64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "count": int(count),
            "estimated_weight": float(estimated_weight),
            "avg_confidence": float(avg_confidence),
            "output_image": f"data:image/jpeg;base64,{output_b64}",
            "detections": detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-stream")
async def detect_stream(image_b64: str):
    # Same as /detect but ensure detections included (already in /detect)
    return await detect(image_b64)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

