from typing import List, Optional
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import UniqueConstraint

class Dataset(SQLModel, table=True):
    __tablename__ = "dataset"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    image_count: int = Field(default=0)
    size: str = Field(default="0 B")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    thumbnail: Optional[str] = Field(default=None, nullable=True)
    
    images: List["DatasetImage"] = Relationship(
        back_populates="dataset",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )

class DatasetImage(SQLModel, table=True):
    __tablename__ = "dataset_image"
    __table_args__ = (
        UniqueConstraint("dataset_id", "filename", name="uq_dataset_image_filename"),
    )
    
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(index=True)
    is_anomaly: Optional[bool] = Field(default=None, nullable=True)
    dataset_id: int = Field(foreign_key="dataset.id", ondelete="CASCADE")
    
    dataset: Dataset = Relationship(back_populates="images")
