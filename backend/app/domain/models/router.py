from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
import os
import re
import time
import logging
import traceback
from engine.model_manager import model_manager

logger = logging.getLogger("ads")
router = APIRouter(prefix="/api/models", tags=["models"])

def sanitize_filename(name):
    """Convert filename to safe ASCII characters."""
    base = os.path.splitext(name)[0]
    safe = re.sub(r'[^\w\-]', '_', base)
    safe = re.sub(r'_+', '_', safe)
    safe = safe.strip('_')
    if not safe:
        safe = f"model_{int(time.time())}"
    return safe

@router.get("")
def list_models():
    """List all models"""
    return {
        "models": model_manager.list_models(),
        "models_detail": model_manager.list_models_detail(),
        "current_model": model_manager.current_model_name,
    }

@router.post("/load")
def load_model(model_name: str = Query(..., description="Name of the model to load")):
    """Load a model"""
    try:
        model_manager.load_model(model_name)
        return {"status": "success", "message": f"Loaded model {model_name}"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a new model file"""
    if not file.filename.endswith(".pth"):
        raise HTTPException(status_code=400, detail="Only .pth files are allowed")
    
    models_dir = getattr(model_manager, "MODELS_DIR", "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    file_path = os.path.join(models_dir, file.filename)
    temp_file_path = file_path + ".tmp"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)
        
        # safely replace the file when upload is complete
        os.replace(temp_file_path, file_path)
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
        
    return {"status": "success", "message": f"Uploaded model {file.filename}"}

@router.delete("/{name}")
def delete_model(name: str):
    """Delete a model file"""
    models_dir = getattr(model_manager, "MODELS_DIR", "models")
    file_path = os.path.join(models_dir, name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    # If this model is currently loaded, unload it
    if model_manager.current_model_name == name:
        model_manager.current_model = None
        model_manager.current_model_name = None
        logger.info(f"Unloaded active model '{name}' before deletion")
    
    try:
        os.remove(file_path)
        logger.info(f"Model '{name}' deleted successfully")
        return {"status": "success", "message": f"Model '{name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
