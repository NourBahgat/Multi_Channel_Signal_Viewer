// ecgService.js
// ---------------
// Provides functions to call backend ECG endpoints:
// - uploadFile
// - getChannelData
// - getPlots
// frontend/services/ecgService.jsx
// frontend/services/ecgService.jsx
import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000/api";

export const fetchEcgCycles = async (recordNumber, leads) => {
  try {
    const params = new URLSearchParams();
    params.append("record_number", recordNumber);
    leads.forEach(lead => params.append("leads", lead));

    const response = await axios.get(`${API_BASE_URL}/ecg`, { params });
    return response.data;
  } catch (error) {
    console.error("Error fetching ECG cycles:", error);
    throw error;
  }
};
