"""
main.py
--------
FastAPI backend entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize app
app = FastAPI()

# ------------------------
# CORS configuration
# ------------------------
origins = ["http://localhost:5173"]  # Vite frontend

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Import routers (relative import)
# ------------------------
from .routers import ecg
app.include_router(ecg.router, prefix="/api/ecg")

# Root endpoint
@app.get("/")
def root():
    return {"message": "Signal Viewer Backend - Ready"}
