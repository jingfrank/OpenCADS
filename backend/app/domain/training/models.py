from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class TrainingJob(SQLModel, table=True):
    __tablename__ = "training_job"
    
    id: str = Field(primary_key=True)  # Celery task UUID / ID
    state: str = Field(default="idle")
    progress: int = Field(default=0)
    stage: str = Field(default="")
    model_name: str = Field(default="")
    dataset_name: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None, nullable=True)
    finished_at: Optional[datetime] = Field(default=None, nullable=True)
    error: Optional[str] = Field(default=None, nullable=True)
