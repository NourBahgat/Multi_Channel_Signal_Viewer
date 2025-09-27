# backend/routers/ecg.py
from fastapi import APIRouter, HTTPException, Query
from typing import List
import wfdb
import numpy as np
from scipy.signal import find_peaks
import os

router = APIRouter()

LUDB_FOLDER = os.path.join(os.path.dirname(__file__), "../services/data")

# -----------------------
# DSP / ECG Processing
# -----------------------
def load_ecg_record(record_number: str):
    """
    Load ECG record from LUDB folder by record_number (e.g., "98").
    """
    record_path = os.path.join(LUDB_FOLDER, f"{record_number}")
    if not os.path.exists(record_path + ".iii"):  # check one lead file exists
        raise FileNotFoundError(f"Record {record_number} not found in {LUDB_FOLDER}")
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal  # shape (num_samples, 12)
    fs = record.fs
    return signals, fs

def get_r_peaks(signal, fs, lead=0):
    distance = int(0.6 * fs)
    peaks, _ = find_peaks(
        signal[:, lead],
        distance=distance,
        height=np.mean(signal[:, lead]) + 0.5 * np.std(signal[:, lead])
    )
    return peaks

def extract_cycles(signals, r_peaks, selected_leads=[0,1,2]):
    cycles = []
    for i in range(len(r_peaks)-1):
        start = r_peaks[i]
        end = r_peaks[i+1]
        cycle = signals[start:end, selected_leads]
        cycles.append(cycle.tolist())  # convert to list for JSON serialization
    return cycles

# -----------------------
# API Endpoints
# -----------------------
@router.get("/")
def get_ecg(
    record_number: str = Query(..., description="ECG record number, e.g., '98'"),
    leads: List[int] = Query([0,1,2], description="List of lead indices (0-11)")
):
    try:
        signals, fs = load_ecg_record(record_number)
        r_peaks = get_r_peaks(signals, fs, lead=leads[0])
        cycles = extract_cycles(signals, r_peaks, selected_leads=leads)
        return {"cycles": cycles, "fs": fs}
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
