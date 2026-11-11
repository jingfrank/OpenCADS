import os
import io
import json
import base64
from datetime import datetime
from typing import List, Optional
from PIL import Image
from sqlmodel import Session, select
from app.domain.datasets.models import Dataset, DatasetImage

DATASETS_DIR = "datasets"

def get_folder_size(folder_path: str) -> int:
    """Calculate total size of a folder in bytes"""
    total_size = 0
    if not os.path.exists(folder_path):
        return 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                try:
                    total_size += os.path.getsize(fp)
                except Exception:
                    pass
    return total_size

def format_size(size_bytes: int) -> str:
    """Format size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def get_image_count(folder_path: str) -> int:
    """Count image files in a folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    count = 0
    if not os.path.exists(folder_path):
        return 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in image_extensions:
                count += 1
    return count

def get_thumbnail_base64(folder_path: str, max_size=(100, 100)) -> Optional[str]:
    """Get base64 encoded thumbnail of first image in folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    if not os.path.exists(folder_path):
        return None
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in image_extensions:
                try:
                    img_path = os.path.join(dirpath, f)
                    with Image.open(img_path) as img:
                        img.thumbnail(max_size)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=70)
                        return base64.b64encode(buffer.getvalue()).decode('utf-8')
                except Exception:
                    continue
    return None

def sync_all_datasets_to_db(db: Session):
    """Sync all datasets and images on disk with the database."""
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
        
    disk_datasets = []
    for name in os.listdir(DATASETS_DIR):
        path = os.path.join(DATASETS_DIR, name)
        if os.path.isdir(path):
            disk_datasets.append(name)
            
    # Remove datasets from DB that are no longer on disk
    stmt = select(Dataset)
    db_datasets = db.exec(stmt).all()
    for db_ds in db_datasets:
        if db_ds.name not in disk_datasets:
            db.delete(db_ds)
    db.commit()
    
    # Sync each dataset on disk
    for name in disk_datasets:
        sync_dataset_metadata(db, name)

def sync_dataset_metadata(db: Session, dataset_name: str):
    """Sync a specific dataset and its images metadata to the database."""
    folder_path = os.path.join(DATASETS_DIR, dataset_name)
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return
        
    stat = os.stat(folder_path)
    image_count = get_image_count(folder_path)
    size_str = format_size(get_folder_size(folder_path))
    created_at = datetime.fromtimestamp(stat.st_ctime)
    thumbnail = get_thumbnail_base64(folder_path)
    
    # Get or create Dataset
    stmt = select(Dataset).where(Dataset.name == dataset_name)
    dataset = db.exec(stmt).first()
    if not dataset:
        dataset = Dataset(
            name=dataset_name,
            image_count=image_count,
            size=size_str,
            created_at=created_at,
            thumbnail=thumbnail
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
    else:
        dataset.image_count = image_count
        dataset.size = size_str
        dataset.thumbnail = thumbnail
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
    # Read labels.json
    labels_file = os.path.join(folder_path, "labels.json")
    labels = {}
    if os.path.exists(labels_file):
        try:
            with open(labels_file, "r") as f:
                labels = json.load(f)
        except Exception:
            pass
            
    # Find all images on disk
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    disk_images = {}
    for filename in os.listdir(folder_path):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            # Determine anomaly status
            if filename in labels:
                is_anomaly = labels[filename]
            else:
                lower_name = filename.lower()
                if 'anomaly' in lower_name or 'bad' in lower_name:
                    is_anomaly = True
                elif 'normal' in lower_name or 'good' in lower_name:
                    is_anomaly = False
                else:
                    is_anomaly = None
            disk_images[filename] = is_anomaly
            
    # Sync images in database
    stmt = select(DatasetImage).where(DatasetImage.dataset_id == dataset.id)
    db_images = db.exec(stmt).all()
    db_image_filenames = {img.filename: img for img in db_images}
    
    # Remove images from DB that are no longer on disk
    for filename, db_img in db_image_filenames.items():
        if filename not in disk_images:
            db.delete(db_img)
            
    # Add or update images
    for filename, is_anomaly in disk_images.items():
        if filename in db_image_filenames:
            db_img = db_image_filenames[filename]
            if db_img.is_anomaly != is_anomaly:
                db_img.is_anomaly = is_anomaly
                db.add(db_img)
        else:
            db_img = DatasetImage(
                filename=filename,
                is_anomaly=is_anomaly,
                dataset_id=dataset.id
            )
            db.add(db_img)
            
    db.commit()
