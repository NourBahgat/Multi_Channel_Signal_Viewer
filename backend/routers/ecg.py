from fastapi import APIRouter

router = APIRouter()

@router.get("/ecg")
def get_ecg():
    return {"status": "ECG endpoint ready"}
