# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from model import AudioCNN
from utils import feature_extract

# Initialize FastAPI app
app = FastAPI(title="Signal Viewer Backend")

# Initialize model
model = AudioCNN()
model.eval()  # set to evaluation mode

# Label mapping
label_map = {0: "Drone", 1: "Bird", 2: "Noise"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Signal Viewer Backend - Ready"}

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        import librosa

        # Load audio file
        arr, sr = librosa.load(file.file, sr=None, mono=True)

        # Extract features
        features = feature_extract(arr, sr)
        features = features.unsqueeze(0)  # add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(features)
            pred_index = int(torch.argmax(output, dim=1).item())

        return JSONResponse({"prediction": label_map[pred_index]})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
