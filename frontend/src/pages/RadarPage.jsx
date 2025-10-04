import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select an audio file");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setPrediction(data.prediction);
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Drone/Bird/Noise Classifier</h1>
      <input type="file" accept="audio/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload & Predict</button>
      {prediction && <h2>Prediction: {prediction}</h2>}
    </div>
  );
}

export default App;
