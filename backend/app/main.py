from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from app.core.database import init_db
from app.domain.datasets.router import router as datasets_router
from app.domain.models.router import router as models_router
from app.domain.training.router import router as training_router
from app.domain.detection.router import router as detection_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database on startup
    init_db()
    yield

app = FastAPI(title="Chip Anomaly Detection System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include domain routers
app.include_router(datasets_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(detection_router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Chip Anomaly Detection System Backend is running"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "System Operational"}

@app.get("/api/system/logs")
def get_system_logs(lines: int = Query(50)):
    """Fetch the latest backend log entries"""
    # The log file is located in the root directory relative to the backend folder
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "backend.log")
    if not os.path.exists(log_file_path):
        # Fallback to local sibling path
        log_file_path = os.path.join("..", "backend.log")
        
    if not os.path.exists(log_file_path):
         return {"logs": ["System log file not found."]}

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            last_lines = [line.strip() for line in last_lines]
            return {"logs": last_lines}
    except Exception as e:
        return {"logs": [f"Error reading logs: {str(e)}"]}
