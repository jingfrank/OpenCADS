from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from sqlmodel import Session, select
import os
import shutil
import zipfile
import json
import logging
from datetime import datetime
from typing import Optional

from app.core.database import get_session
from app.domain.datasets.models import Dataset, DatasetImage
from app.domain.datasets.services import (
    DATASETS_DIR,
    sync_all_datasets_to_db,
    sync_dataset_metadata,
)

logger = logging.getLogger("ads")
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.get("")
def list_datasets(db: Session = Depends(get_session)):
    """List all datasets with metadata from the database"""
    sync_all_datasets_to_db(db)
    
    stmt = select(Dataset)
    datasets = db.exec(stmt).all()
    
    # Format according to expectations
    result = []
    for d in datasets:
        result.append({
            "name": d.name,
            "image_count": d.image_count,
            "size": d.size,
            "created_at": d.created_at.strftime("%Y-%m-%d %H:%M"),
            "thumbnail": d.thumbnail
        })
        
    # Sort by creation time, newest first
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return {"datasets": result}

@router.post("")
async def create_dataset(
    category: str,
    label: str = Query(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_session)
):
    """Create a new dataset by uploading images or a zip file"""
    # Sanitize category name
    safe_name = "".join(c for c in category if c.isalnum() or c in ('_', '-', ' ')).strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid category name")
    
    dataset_path = os.path.join(DATASETS_DIR, safe_name)
    
    # Create dataset folder if not exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logger.info(f"Created new dataset directory: {dataset_path}")
    
    filename = file.filename.lower()
    logger.info(f"Uploading file '{filename}' to dataset '{safe_name}'")
    
    extracted_files = []
    
    if filename.endswith('.zip'):
        # Handle zip file
        temp_zip = os.path.join(DATASETS_DIR, f"temp_{file.filename}")
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
                extracted_files = zip_ref.namelist()
        except zipfile.BadZipFile:
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            raise HTTPException(status_code=400, detail="Invalid zip file")
        finally:
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
    
    elif filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')):
        # Handle single image
        file_path = os.path.join(dataset_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        extracted_files = [file.filename]
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use ZIP or image files (JPG, PNG, etc.)")
    
    if label in ("normal", "anomaly"):
        is_anomaly = (label == "anomaly")
        labels_file = os.path.join(dataset_path, "labels.json")
        labels = {}
        if os.path.exists(labels_file):
            try:
                with open(labels_file, "r") as f:
                    labels = json.load(f)
            except Exception:
                pass
                
        # Update labels for extracted images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        for f in extracted_files:
            base_f = os.path.basename(f)
            if os.path.splitext(base_f)[1].lower() in image_extensions:
                labels[base_f] = is_anomaly
                
        with open(labels_file, "w") as f:
            json.dump(labels, f)
            
    # Sync metadata to database
    sync_dataset_metadata(db, safe_name)
    
    stmt = select(Dataset).where(Dataset.name == safe_name)
    dataset = db.exec(stmt).first()
    if not dataset:
        raise HTTPException(status_code=500, detail="Failed to sync created dataset")
        
    return {
        "status": "success", 
        "message": f"Dataset '{safe_name}' created/updated",
        "dataset": {
            "name": dataset.name,
            "image_count": dataset.image_count,
            "size": dataset.size,
            "created_at": dataset.created_at.strftime("%Y-%m-%d %H:%M"),
            "thumbnail": dataset.thumbnail
        }
    }

@router.delete("/{name}")
def delete_dataset(name: str, db: Session = Depends(get_session)):
    """Delete a dataset"""
    dataset_path = os.path.join(DATASETS_DIR, name)
    
    stmt = select(Dataset).where(Dataset.name == name)
    dataset = db.exec(stmt).first()
    
    if not os.path.exists(dataset_path) and not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    if os.path.exists(dataset_path):
        try:
            shutil.rmtree(dataset_path)
            logger.info(f"Dataset '{name}' directory deleted successfully")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    if dataset:
        db.delete(dataset)
        db.commit()
        logger.info(f"Dataset '{name}' record deleted from database")
        
    return {"status": "success", "message": f"Dataset '{name}' deleted"}

@router.get("/{name}/images")
def list_dataset_images(name: str, db: Session = Depends(get_session)):
    """List images in a specific dataset from the database"""
    # Ensure the metadata is synced in db
    sync_dataset_metadata(db, name)
    
    stmt = select(Dataset).where(Dataset.name == name)
    dataset = db.exec(stmt).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    stmt_img = select(DatasetImage).where(DatasetImage.dataset_id == dataset.id)
    db_images = db.exec(stmt_img).all()
    
    images = []
    for img in db_images:
        images.append({
            "id": img.filename,
            "filename": img.filename,
            "is_anomaly": img.is_anomaly
        })
        
    images.sort(key=lambda x: x["filename"])
    return {"images": images}

@router.post("/{name}/images/{filename}/label")
async def label_dataset_image(
    name: str,
    filename: str,
    is_anomaly: Optional[bool] = Query(None),
    db: Session = Depends(get_session)
):
    """Label or unlabel an image in the dataset"""
    dataset_path = os.path.join(DATASETS_DIR, name)
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    labels_file = os.path.join(dataset_path, "labels.json")
    labels = {}
    if os.path.exists(labels_file):
        try:
            with open(labels_file, "r") as f:
                labels = json.load(f)
        except Exception:
            pass
            
    if is_anomaly is None:
        labels.pop(filename, None)
    else:
        labels[filename] = is_anomaly
        
    with open(labels_file, "w") as f:
        json.dump(labels, f)
        
    # Sync metadata to database
    sync_dataset_metadata(db, name)
    
    # Check if the image exists in DB now (it should have been synced if it exists on disk)
    stmt = select(Dataset).where(Dataset.name == name)
    dataset = db.exec(stmt).first()
    if dataset:
        stmt_img = select(DatasetImage).where(DatasetImage.dataset_id == dataset.id, DatasetImage.filename == filename)
        db_image = db.exec(stmt_img).first()
        if not db_image:
            # If the file wasn't on disk (and thus not synced), raise 404?
            # Wait, the original endpoint did not raise 404, but let's be robust
            pass
            
    return {"status": "success"}

@router.get("/{name}/images/{filename}")
def get_dataset_image(name: str, filename: str):
    """Serve an image from a dataset"""
    file_path = os.path.join(DATASETS_DIR, name, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)
