from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select
import os
import time
import logging
from datetime import datetime

from app.core.database import get_session
from app.domain.training.models import TrainingJob
from app.domain.datasets.services import DATASETS_DIR
from app.domain.training.services import train_with_status_task
from app.domain.models.router import sanitize_filename

logger = logging.getLogger("ads")
router = APIRouter(prefix="/api/train", tags=["training"])

@router.post("")
async def train(
    background_tasks: BackgroundTasks,
    data: dict,
    db: Session = Depends(get_session)
):
    """Start model training using a dataset from the 'datasets' folder"""
    # Check if already training
    stmt = select(TrainingJob).where(TrainingJob.state == "running")
    running_job = db.exec(stmt).first()
    if running_job:
        raise HTTPException(status_code=409, detail="Training is already in progress")

    dataset_name = data.get("dataset_name")
    if not dataset_name:
        raise HTTPException(status_code=400, detail="dataset_name is required")
        
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    if not os.path.exists(dataset_path):
         raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")

    # Extract PatchCore parameters
    backbone = data.get("backbone", "wide_resnet50_2")
    coreset_ratio = float(data.get("coreset_ratio", 0.1))
    n_neighbors = int(data.get("n_neighbors", 9))
    
    # Generate model name
    safe_name = sanitize_filename(dataset_name)
    model_name = f"{safe_name}_model.pth"
    
    job_id = str(int(time.time() * 1000))
    
    # Initialize training status in DB
    job = TrainingJob(
        id=job_id,
        state="running",
        progress=0,
        stage="初始化训练环境",
        model_name=model_name,
        dataset_name=dataset_name,
        started_at=datetime.now(),
        finished_at=None,
        error=None
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Trigger training in background
    background_tasks.add_task(
        train_with_status_task,
        job_id=job_id,
        dataset_path=dataset_path,
        model_name=model_name,
        backbone=backbone,
        coreset_ratio=coreset_ratio,
        n_neighbors=n_neighbors
    )
    
    logger.info(f"Training started for dataset '{dataset_name}', job ID: {job_id}")
    return {
        "status": "success", 
        "message": f"Training started for dataset '{dataset_name}'", 
        "model_name": model_name
    }

@router.get("/status")
def get_training_status(db: Session = Depends(get_session)):
    """Get current training status"""
    # Find running job first
    stmt = select(TrainingJob).where(TrainingJob.state == "running")
    job = db.exec(stmt).first()
    
    # If not running, find the most recently started job
    if not job:
        stmt_recent = select(TrainingJob).order_by(TrainingJob.started_at.desc())
        job = db.exec(stmt_recent).first()
        
    if not job:
        return {
            "state": "idle",
            "progress": 0,
            "stage": "",
            "model_name": "",
            "dataset_name": "",
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
        
    return {
        "state": job.state,
        "progress": job.progress,
        "stage": job.stage,
        "model_name": job.model_name,
        "dataset_name": job.dataset_name,
        "started_at": job.started_at.strftime("%Y-%m-%d %H:%M:%S") if job.started_at else None,
        "finished_at": job.finished_at.strftime("%Y-%m-%d %H:%M:%S") if job.finished_at else None,
        "error": job.error,
    }
