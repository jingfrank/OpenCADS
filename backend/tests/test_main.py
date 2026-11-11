import os
import io
import zipfile
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock
from PIL import Image
import app.main
import engine.model_manager

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Chip Anomaly Detection System Backend is running"}

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "System Operational"}

def test_list_models(client, tmp_path):
    # Write a mock model file
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "test_model_1.pth"
    
    checkpoint = {
        "init_params": {
            "backbone_name": "resnet18",
            "precentage": 0.1,
            "n_nearest_neighbour": 3,
            "target_dim": 1024
        },
        "threshold": 0.5,
        "train_score_mean": 0.1,
        "train_score_std": 0.02,
        "memory_bank_state": {
            "bank": np.zeros((10, 1024))
        }
    }
    torch.save(checkpoint, model_path)
    
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "test_model_1.pth" in data["models"]
    assert len(data["models_detail"]) == 1
    assert data["models_detail"][0]["name"] == "test_model_1.pth"
    assert data["models_detail"][0]["backbone"] == "resnet18"
    assert data["current_model"] is None

def test_load_model(client, tmp_path):
    # Non-existent model
    response = client.post("/api/models/load?model_name=nonexistent.pth")
    assert response.status_code == 400
    
    # Existing model
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save({"init_params": {}}, models_dir / "test_model_1.pth")
    
    response = client.post("/api/models/load?model_name=test_model_1.pth")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_train_and_status(client, tmp_path):
    # Missing dataset_name
    response = client.post("/api/train", json={})
    assert response.status_code == 400
    
    # Nonexistent dataset
    response = client.post("/api/train", json={"dataset_name": "nonexistent"})
    assert response.status_code == 404
    
    # Create dummy dataset directory
    datasets_dir = tmp_path / "datasets"
    dataset_dir = datasets_dir / "dummy_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy image
    img = Image.new("RGB", (100, 100))
    img.save(dataset_dir / "img1.png")
    
    # Successful training request (background task executes synchronously in TestClient)
    response = client.post("/api/train", json={"dataset_name": "dummy_dataset"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model_name"] == "dummy_dataset_model.pth"
    
    # Check status - should be finished because TestClient runs tasks synchronously
    status_response = client.get("/api/train/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["state"] == "finished"
    assert status_data["progress"] == 100
    assert (tmp_path / "models" / "dummy_dataset_model.pth").exists()
    
    # Already training conflict (manually mock running state in DB)
    from sqlmodel import Session
    from app.core.database import engine
    from app.domain.training.models import TrainingJob
    from datetime import datetime
    
    with Session(engine) as session:
        job = TrainingJob(
            id="mock_running_job",
            state="running",
            progress=50,
            stage="训练中",
            model_name="dummy_dataset_model.pth",
            dataset_name="dummy_dataset",
            started_at=datetime.now()
        )
        session.add(job)
        session.commit()
        
    response = client.post("/api/train", json={"dataset_name": "dummy_dataset"})
    assert response.status_code == 409

def test_train_error(client, tmp_path, monkeypatch):
    datasets_dir = tmp_path / "datasets"
    dataset_dir = datasets_dir / "dummy_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100)).save(dataset_dir / "img1.png")
    
    # Mock train_model to raise exception
    def mock_raise_train(*args, **kwargs):
        raise ValueError("Simulated training error")
    import app.domain.training.services
    monkeypatch.setattr(app.domain.training.services, "train_model", mock_raise_train)
    
    response = client.post("/api/train", json={"dataset_name": "dummy_dataset"})
    assert response.status_code == 200  # Starts background task
    
    # Check status - should be error
    status_response = client.get("/api/train/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["state"] == "error"
    assert "ValueError" in status_data["error"] or "Simulated training error" in status_data["error"]

def test_upload_model(client, tmp_path):
    # Invalid extension
    response = client.post(
        "/api/models/upload", 
        files={"file": ("model.txt", b"some data", "text/plain")}
    )
    assert response.status_code == 400
    
    # Valid extension
    response = client.post(
        "/api/models/upload", 
        files={"file": ("uploaded_model.pth", b"some pth data", "application/octet-stream")}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert (tmp_path / "models" / "uploaded_model.pth").exists()

def test_delete_model(client, tmp_path):
    # Non-existent
    response = client.delete("/api/models/nonexistent.pth")
    assert response.status_code == 404
    
    # Create dummy models
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_file_1 = models_dir / "delete_me.pth"
    model_file_1.write_bytes(b"dummy")
    
    # Setup active model name
    engine.model_manager.model_manager.current_model = MagicMock()
    engine.model_manager.model_manager.current_model_name = "delete_me.pth"
    
    response = client.delete("/api/models/delete_me.pth")
    assert response.status_code == 200
    assert not model_file_1.exists()
    assert engine.model_manager.model_manager.current_model_name is None
    assert engine.model_manager.model_manager.current_model is None

def test_predict_endpoints_and_history(client, tmp_path):
    # 1. Predict without model loaded
    response = client.post(
        "/api/predict", 
        files={"file": ("test.png", b"fake_png_data", "image/png")}
    )
    assert response.status_code == 400
    
    # 2. Load model (existing model mock)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save({"init_params": {}}, models_dir / "pred_model.pth")
    client.post("/api/models/load?model_name=pred_model.pth")
    
    # 3. Predict with model
    response = client.post(
        "/api/predict", 
        files={"file": ("test.png", b"fake_png_data", "image/png")},
        params={"dataset_name": "demo"}
    )
    assert response.status_code == 200
    pred_data = response.json()
    assert "score" in pred_data
    assert "heatmap" in pred_data
    assert "mask" in pred_data
    
    # 4. Check history
    history_response = client.get("/api/detection/history?limit=10")
    assert history_response.status_code == 200
    history = history_response.json()["history"]
    assert len(history) == 1
    item_id = history[0]["id"]
    assert history[0]["filename"] == "test.png"
    assert history[0]["dataset_name"] == "demo"
    
    # 5. Check heatmap and mask downloads
    heatmap_res = client.get(f"/api/detection/history/{item_id}/heatmap")
    assert heatmap_res.status_code == 200
    
    mask_res = client.get(f"/api/detection/history/{item_id}/mask")
    assert mask_res.status_code == 200
    
    # Non-existent heatmap/mask
    assert client.get("/api/detection/history/nonexistent/heatmap").status_code == 404
    assert client.get("/api/detection/history/nonexistent/mask").status_code == 404
    
    # 6. Post QC Review
    qc_res = client.post(
        f"/api/detection/history/{item_id}/qc", 
        json={"result": "ng", "notes": "defect found", "reviewer": "qa_operator"}
    )
    assert qc_res.status_code == 200
    assert qc_res.json()["item"]["qc_status"] == "reviewed"
    assert qc_res.json()["item"]["qc_result"] == "ng"
    
    # Non-existent QC
    assert client.post("/api/detection/history/nonexistent/qc", json={"result": "ok"}).status_code == 404
    
    # 7. Clear history
    clear_res = client.delete("/api/detection/history")
    assert clear_res.status_code == 200
    assert len(client.get("/api/detection/history").json()["history"]) == 0
    
    # Verify files are deleted
    assert not (tmp_path / "uploads" / "results" / f"{item_id}_heatmap.png").exists()

def test_datasets_management(client, tmp_path):
    # 1. List datasets (empty)
    res = client.get("/api/datasets")
    assert res.status_code == 200
    assert len(res.json()["datasets"]) == 0
    
    # 2. POST /api/datasets (invalid category)
    res = client.post("/api/datasets?category=...", files={"file": ("img.png", b"dummy", "image/png")})
    assert res.status_code == 400
    
    # 3. POST /api/datasets (unsupported type)
    res = client.post("/api/datasets?category=test_cat", files={"file": ("img.txt", b"dummy", "text/plain")})
    assert res.status_code == 400
    
    # 4. POST /api/datasets (single image, label=anomaly)
    img_io = io.BytesIO()
    Image.new("RGB", (10, 10)).save(img_io, format="PNG")
    img_bytes = img_io.getvalue()
    
    res = client.post(
        "/api/datasets?category=cat_single&label=anomaly", 
        files={"file": ("image_one.png", img_bytes, "image/png")}
    )
    assert res.status_code == 200
    assert res.json()["dataset"]["name"] == "cat_single"
    assert res.json()["dataset"]["image_count"] == 1
    
    # 5. POST /api/datasets (zip file, label=normal)
    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w") as zf:
        zf.writestr("normal_img1.png", img_bytes)
        zf.writestr("normal_img2.png", img_bytes)
    zip_bytes = zip_io.getvalue()
    
    res = client.post(
        "/api/datasets?category=cat_zip&label=normal", 
        files={"file": ("archive.zip", zip_bytes, "application/zip")}
    )
    assert res.status_code == 200
    assert res.json()["dataset"]["name"] == "cat_zip"
    assert res.json()["dataset"]["image_count"] == 2
    
    # 6. List datasets (should have three datasets: cat_single, cat_zip, and test_cat from the bad upload test)
    res = client.get("/api/datasets")
    assert res.status_code == 200
    assert len(res.json()["datasets"]) == 3
    
    # 7. List dataset images
    res = client.get("/api/datasets/cat_zip/images")
    assert res.status_code == 200
    images = res.json()["images"]
    assert len(images) == 2
    assert images[0]["is_anomaly"] is False
    
    # Non-existent dataset images
    assert client.get("/api/datasets/nonexistent/images").status_code == 404
    
    # 8. Serve dataset image
    img_res = client.get("/api/datasets/cat_zip/images/normal_img1.png")
    assert img_res.status_code == 200
    
    # Serve nonexistent image
    assert client.get("/api/datasets/cat_zip/images/nonexistent.png").status_code == 404
    
    # 9. Label dataset image
    label_res = client.post("/api/datasets/cat_zip/images/normal_img1.png/label?is_anomaly=true")
    assert label_res.status_code == 200
    
    # Verify updated label
    res = client.get("/api/datasets/cat_zip/images")
    images = res.json()["images"]
    for img_info in images:
        if img_info["filename"] == "normal_img1.png":
            assert img_info["is_anomaly"] is True
            
    # Remove label
    label_res = client.post("/api/datasets/cat_zip/images/normal_img1.png/label")
    assert label_res.status_code == 200
    
    # Verify label removed (should be None or not present in labels)
    res = client.get("/api/datasets/cat_zip/images")
    images = res.json()["images"]
    for img_info in images:
        if img_info["filename"] == "normal_img1.png":
            # Since filename is normal_img1.png, it auto-detects is_anomaly as False
            assert img_info["is_anomaly"] is False
    
    # Non-existent dataset label
    assert client.post("/api/datasets/nonexistent/images/img.png/label?is_anomaly=true").status_code == 404
    
    # 10. Delete dataset
    del_res = client.delete("/api/datasets/cat_single")
    assert del_res.status_code == 200
    assert not (tmp_path / "datasets" / "cat_single").exists()
    
    # Delete nonexistent dataset
    assert client.delete("/api/datasets/nonexistent").status_code == 404

def test_system_logs(client):
    res = client.get("/api/system/logs?lines=10")
    assert res.status_code == 200
    assert "logs" in res.json()
    assert isinstance(res.json()["logs"], list)
