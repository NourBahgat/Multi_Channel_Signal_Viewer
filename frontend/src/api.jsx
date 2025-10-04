import React, { useState, useRef } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    // نعمل URL محلي للملف علشان نشغله
    if (selectedFile) {
      setAudioUrl(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a file first!");

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Something went wrong!");
    }
    setLoading(false);
  };

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play();
      setIsPlaying(true);
    }

    // لما يخلص الصوت يرجع الزرار لحالة الإيقاف
    audio.onended = () => setIsPlaying(false);
  };

  return (
    <div style={{ textAlign: "center", marginTop: "40px" }}>
      <h2>Drone / Noise Detection</h2>

      <input type="file" accept=".wav" onChange={handleFileChange} />
      <button
        onClick={handleUpload}
        disabled={loading}
        style={{ marginLeft: "10px" }}
      >
        {loading ? "Processing..." : "Upload & Predict"}
      </button>

      {audioUrl && (
        <div style={{ marginTop: "20px" }}>
          <h3>Uploaded Audio</h3>
          <audio ref={audioRef} src={audioUrl} />
          <button
            onClick={togglePlay}
            style={{
              marginTop: "10px",
              padding: "8px 20px",
              fontSize: "16px",
              cursor: "pointer",
            }}
          >
            {isPlaying ? "⏸ Pause" : "▶️ Play"}
          </button>
        </div>
      )}

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Prediction Result</h3>
          <p>
            <strong>Predicted Label:</strong> {result.predicted_label}
          </p>
          <p>
            <strong>Confidence:</strong>{" "}
             {typeof result.confidence === "number"
              ? `${result.confidence.toFixed(2)}%`
              : result.confidence}
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
