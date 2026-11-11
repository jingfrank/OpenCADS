from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class DetectionHistory(SQLModel, table=True):
    __tablename__ = "detection_history"
    
    id: str = Field(primary_key=True)
    filename: str
    dataset_name: str = Field(default="")
    score: float
    threshold: float
    is_anomaly: bool
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # QC fields
    qc_status: str = Field(default="pending")
    qc_result: Optional[str] = Field(default=None, nullable=True)
    qc_notes: str = Field(default="")
    qc_reviewer: str = Field(default="")
    qc_timestamp: Optional[datetime] = Field(default=None, nullable=True)
