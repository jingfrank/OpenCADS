from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session
import os
from typing import Optional

from app.core.database import get_session
from app.domain.detection.services import (
    RESULTS_DIR,
    handle_prediction,
    get_history,
    submit_qc,
    clear_history,
)

router = APIRouter(prefix="/api", tags=["detection"])

class QcRequest(BaseModel):
    result: str = "ok"
    notes: str = ""
    reviewer: str = ""

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = None,
    db: Session = Depends(get_session)
):
    """Predict if an image is anomalous using the loaded model"""
    try:
        result = handle_prediction(db, file.file, file.filename, dataset_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/detection/history")
def get_detection_history(limit: int = Query(50), db: Session = Depends(get_session)):
    """Get detection history list"""
    history = get_history(db, limit)
    
    formatted = []
    for item in history:
        formatted.append({
            "id": item.id,
            "filename": item.filename,
            "dataset_name": item.dataset_name,
            "score": item.score,
            "threshold": item.threshold,
            "is_anomaly": item.is_anomaly,
            "model": item.model,
            "timestamp": item.timestamp.strftime("%Y-%m-%d %H:%M:%S") if item.timestamp else "",
            "qc_status": item.qc_status,
            "qc_result": item.qc_result,
            "qc_notes": item.qc_notes,
            "qc_reviewer": item.qc_reviewer,
            "qc_timestamp": item.qc_timestamp.strftime("%Y-%m-%d %H:%M:%S") if item.qc_timestamp else ""
        })
    return {"history": formatted}

@router.get("/detection/history/{item_id}/heatmap")
def get_history_heatmap(item_id: str):
    """Serve the heatmap image for a history item"""
    file_path = os.path.join(RESULTS_DIR, f"{item_id}_heatmap.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(file_path)

@router.get("/detection/history/{item_id}/mask")
def get_history_mask(item_id: str):
    """Serve the mask image for a history item"""
    file_path = os.path.join(RESULTS_DIR, f"{item_id}_mask.png")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Mask not found")
    return FileResponse(file_path)

@router.post("/detection/history/{item_id}/qc")
def submit_qc_review(item_id: str, qc: QcRequest, db: Session = Depends(get_session)):
    """Submit QC review for a detection history item"""
    item = submit_qc(db, item_id, qc.result, qc.notes, qc.reviewer)
    if not item:
        raise HTTPException(status_code=404, detail="History record not found")
        
    return {
        "status": "success",
        "message": "QC submitted",
        "item": {
            "id": item.id,
            "filename": item.filename,
            "dataset_name": item.dataset_name,
            "score": item.score,
            "threshold": item.threshold,
            "is_anomaly": item.is_anomaly,
            "model": item.model,
            "timestamp": item.timestamp.strftime("%Y-%m-%d %H:%M:%S") if item.timestamp else "",
            "qc_status": item.qc_status,
            "qc_result": item.qc_result,
            "qc_notes": item.qc_notes,
            "qc_reviewer": item.qc_reviewer,
            "qc_timestamp": item.qc_timestamp.strftime("%Y-%m-%d %H:%M:%S") if item.qc_timestamp else ""
        }
    }

@router.delete("/detection/history")
def clear_detection_history(db: Session = Depends(get_session)):
    """Clear all detection history"""
    clear_history(db)
    return {"status": "success", "message": "History cleared"}
