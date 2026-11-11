from datetime import datetime
import logging
import traceback
from sqlmodel import Session, select
from app.core.database import engine
from app.domain.training.models import TrainingJob
from engine.trainer import train_model

logger = logging.getLogger("ads")

def make_progress_callback(job_id: str):
    """Callback generator to report progress of a training job to the DB"""
    def training_progress_callback(stage: str, progress: int):
        with Session(engine) as session:
            stmt = select(TrainingJob).where(TrainingJob.id == job_id)
            job = session.exec(stmt).first()
            if job:
                job.stage = stage
                job.progress = progress
                session.add(job)
                session.commit()
        logger.info(f"Training progress for job {job_id}: {stage} ({progress}%)")
    return training_progress_callback

def train_with_status_task(
    job_id: str,
    dataset_path: str,
    model_name: str,
    backbone: str,
    coreset_ratio: float,
    n_neighbors: int
):
    """Background task to run model training and update the DB job status"""
    callback = make_progress_callback(job_id)
    try:
        train_model(
            dataset_path,
            model_name=model_name,
            backbone_name=backbone,
            coreset_percentage=coreset_ratio,
            n_nearest_neighbour=n_neighbors,
            progress_callback=callback
        )
        with Session(engine) as session:
            stmt = select(TrainingJob).where(TrainingJob.id == job_id)
            job = session.exec(stmt).first()
            if job:
                job.state = "finished"
                job.progress = 100
                job.stage = "训练完成"
                job.finished_at = datetime.now()
                session.add(job)
                session.commit()
        logger.info(f"Training job {job_id} completed successfully")
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        traceback.print_exc()
        with Session(engine) as session:
            stmt = select(TrainingJob).where(TrainingJob.id == job_id)
            job = session.exec(stmt).first()
            if job:
                job.state = "error"
                job.stage = f"训练失败: {str(e)}"
                job.error = str(e)
                job.finished_at = datetime.now()
                session.add(job)
                session.commit()
