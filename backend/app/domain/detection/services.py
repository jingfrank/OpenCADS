import os
import time
import base64
import logging
from datetime import datetime
from typing import Optional, List
from sqlmodel import Session, select
from app.domain.detection.models import DetectionHistory
from engine.inference import predict_image
from engine.model_manager import model_manager

logger = logging.getLogger("ads")

UPLOAD_DIR = "uploads"
RESULTS_DIR = os.path.join(UPLOAD_DIR, "results")
MAX_HISTORY = 500

def handle_prediction(
    db: Session,
    image_file,
    filename: str,
    dataset_name: Optional[str] = None
) -> dict:
    """Run model inference and save the result to database and disk files"""
    if model_manager.get_model() is None:
        raise ValueError("No model loaded. Please load or train a model first.")
        
    # Run prediction
    result = predict_image(image_file)
    
    # Generate unique ID based on time
    history_id = str(int(time.time() * 1000))
    
    # Persist prediction images
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    if "heatmap" in result and result["heatmap"]:
        with open(os.path.join(RESULTS_DIR, f"{history_id}_heatmap.png"), "wb") as f:
            f.write(base64.b64decode(result["heatmap"]))
            
    if "mask" in result and result["mask"]:
        with open(os.path.join(RESULTS_DIR, f"{history_id}_mask.png"), "wb") as f:
            f.write(base64.b64decode(result["mask"]))
            
    # Create database record
    history_entry = DetectionHistory(
        id=history_id,
        filename=filename or "unknown",
        dataset_name=dataset_name or "",
        score=result["score"],
        threshold=result["threshold"],
        is_anomaly=result["is_anomaly"],
        model=model_manager.current_model_name or "unknown",
        timestamp=datetime.now(),
        qc_status="pending",
        qc_result=None,
        qc_notes="",
        qc_reviewer="",
        qc_timestamp=None
    )
    
    db.add(history_entry)
    db.commit()
    db.refresh(history_entry)
    
    # Trim history in database
    stmt = select(DetectionHistory).order_by(DetectionHistory.timestamp.desc())
    records = db.exec(stmt).all()
    if len(records) > MAX_HISTORY:
        for old_rec in records[MAX_HISTORY:]:
            # Delete corresponding files
            for suffix in ["_heatmap.png", "_mask.png"]:
                fpath = os.path.join(RESULTS_DIR, f"{old_rec.id}{suffix}")
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
            db.delete(old_rec)
        db.commit()
        
    return result

def get_history(db: Session, limit: int = 50) -> List[DetectionHistory]:
    """Get detection history list from database"""
    stmt = select(DetectionHistory).order_by(DetectionHistory.timestamp.desc()).limit(limit)
    return db.exec(stmt).all()

def submit_qc(
    db: Session,
    item_id: str,
    result: str,
    notes: str,
    reviewer: str
) -> Optional[DetectionHistory]:
    """Submit QC review for a detection history item"""
    stmt = select(DetectionHistory).where(DetectionHistory.id == item_id)
    item = db.exec(stmt).first()
    if item:
        item.qc_status = "reviewed"
        item.qc_result = result
        item.qc_notes = notes
        item.qc_reviewer = reviewer
        item.qc_timestamp = datetime.now()
        db.add(item)
        db.commit()
        db.refresh(item)
    return item

def clear_history(db: Session):
    """Clear all history from database and files from disk"""
    stmt = select(DetectionHistory)
    records = db.exec(stmt).all()
    for rec in records:
        db.delete(rec)
    db.commit()
    
    if os.path.exists(RESULTS_DIR):
        for fname in os.listdir(RESULTS_DIR):
            fpath = os.path.join(RESULTS_DIR, fname)
            try:
                os.remove(fpath)
            except Exception:
                pass
