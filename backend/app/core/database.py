import os
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import event
from typing import Generator

# Import models so SQLModel metadata registry knows about them for create_all()
from app.domain.datasets.models import Dataset, DatasetImage
from app.domain.detection.models import DetectionHistory
from app.domain.training.models import TrainingJob

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ads.db")

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

def init_db():
    """Create all database tables defined in SQLModel schemas."""
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    """Dependency generator to yield database sessions."""
    with Session(engine) as session:
        yield session
