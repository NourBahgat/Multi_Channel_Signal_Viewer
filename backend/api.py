from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import librosa
import numpy as np
import os
from uuid import uuid4

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------------
# Define model
# -------------------------------
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return torch.tensor(mfcc_scaled, dtype=torch.float32)


# -------------------------------
# Load model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier()
model.load_state_dict(torch.load(
    "F:/Dsp task 1/Multi_Channel_Signal_Viewer/model.pth",
    map_location=device
))
model.to(device)
model.eval()

label_map = {0: "Noise", 1: "Drone"}


# -------------------------------
# Audio streaming endpoint
# -------------------------------
@app.route("/play/<filename>", methods=["GET"])
def play_audio(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="audio/wav")
    else:
        return jsonify({"error": "File not found"}), 404


# -------------------------------
# Prediction API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = f"{uuid4().hex}.wav"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        features = extract_features(save_path).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)[0]  # ensures shape (2,)

        probs_np = probs.cpu().numpy()
        predicted_idx = int(np.argmax(probs_np))
        pred_label = label_map[predicted_idx]
        confidence = float(np.clip(probs_np[predicted_idx], 0.0, 1.0))  # clamp
        confidence = round(confidence * 100, 2)

    except Exception as e:
        os.remove(save_path)
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "predicted_label": pred_label,
        "confidence": confidence,
        "file_url": f"http://127.0.0.1:5000/play/{filename}"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
