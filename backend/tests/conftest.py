import sys
import os
import tempfile
import shutil
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# 1. Setup session-wide temp directory and change CWD
session_temp_dir = tempfile.mkdtemp(prefix="ads_test_session_")
os.chdir(session_temp_dir)

# 2. Add backend dir and its parent to sys.path so we can import modules correctly
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath(os.path.join(backend_dir, ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import top-level modules first
import engine.model_manager
import engine.trainer
import engine.inference
import engine.patchcore
import app.main
import main

# Alias backend modules to top-level modules in sys.modules
sys.modules["backend.engine"] = sys.modules["engine"]
sys.modules["backend.engine.model_manager"] = sys.modules["engine.model_manager"]
sys.modules["backend.engine.trainer"] = sys.modules["engine.trainer"]
sys.modules["backend.engine.inference"] = sys.modules["engine.inference"]
sys.modules["backend.engine.patchcore"] = sys.modules["engine.patchcore"]

sys.modules["backend.app"] = sys.modules["app"]
sys.modules["backend.app.main"] = sys.modules["app.main"]
sys.modules["backend.main"] = sys.modules["main"]

@pytest.fixture(autouse=True)
def isolate_test_fs(monkeypatch, tmp_path):
    # Change current working directory to the test-specific tmp_path
    monkeypatch.chdir(tmp_path)
    
    # Create required directories
    datasets_dir = tmp_path / "datasets"
    upload_dir = tmp_path / "uploads"
    results_dir = upload_dir / "results"
    models_dir = tmp_path / "models"
    
    datasets_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the domain services global paths
    import app.domain.datasets.services
    import app.domain.detection.services
    import app.domain.training.router
    
    monkeypatch.setattr(app.domain.datasets.services, "DATASETS_DIR", str(datasets_dir))
    monkeypatch.setattr(app.domain.detection.services, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(app.domain.detection.services, "RESULTS_DIR", str(results_dir))
    
    # Set the model manager global MODELS_DIR
    monkeypatch.setattr(engine.model_manager, "MODELS_DIR", str(models_dir))
    
    # Setup test database engine
    from sqlmodel import create_engine
    import app.core.database
    import app.domain.training.services
    
    test_db_file = tmp_path / "test_ads.db"
    test_engine = create_engine(f"sqlite:///{test_db_file}", connect_args={"check_same_thread": False})
    
    monkeypatch.setattr(app.core.database, "engine", test_engine)
    monkeypatch.setattr(app.domain.training.services, "engine", test_engine)
    
    # Re-initialize the database tables on the test engine
    app.core.database.init_db()
    
    # Reset model_manager state
    monkeypatch.setattr(engine.model_manager.model_manager, "current_model", None)
    monkeypatch.setattr(engine.model_manager.model_manager, "current_model_name", None)

@pytest.fixture(autouse=True)
def mock_ml_components(monkeypatch):
    import numpy as np
    
    # Mock train_model
    def mock_train_model(data_path, model_name="custom_model.pth", 
                         backbone_name="wide_resnet50_2",
                         layers_to_extract=(2, 3),
                         patch_size=3,
                         stride=1,
                         target_dim=1024,
                         coreset_percentage=1.0,
                         n_nearest_neighbour=3,
                         batch_size=8,
                         image_size=256,
                         progress_callback=None):
        if progress_callback:
            progress_callback("加载数据集", 10)
            progress_callback("特征提取中", 50)
            progress_callback("训练完成", 100)
        
        # Save a dummy checkpoint file
        import torch
        import engine.model_manager
        model_path = os.path.join(engine.model_manager.MODELS_DIR, model_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        checkpoint = {
            "init_params": {
                "backbone_name": backbone_name,
                "precentage": coreset_percentage,
                "n_nearest_neighbour": n_nearest_neighbour,
                "target_dim": target_dim
            },
            "threshold": 0.5,
            "train_score_mean": 0.1,
            "train_score_std": 0.02,
            "memory_bank_state": {
                "bank": np.zeros((5, 128))
            },
            "forward_modules_state": {}
        }
        torch.save(checkpoint, model_path)
    
    # Mock predict_image
    def mock_predict_image(image_file, threshold=0.5):
        # Return a dummy prediction dict
        # dummy base64 1x1 png image
        dummy_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        return {
            "score": 0.15,
            "heatmap": dummy_png,
            "mask": dummy_png,
            "is_anomaly": False,
            "threshold": threshold
        }
    
    # Mock load_model
    def mock_load_model(model_name):
        import engine.model_manager
        path = os.path.join(engine.model_manager.MODELS_DIR, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {model_name} not found")
        
        mock_model = MagicMock()
        mock_model.threshold = 0.5
        mock_model.predict = MagicMock(return_value=(0.15, np.zeros((1, 256, 256))))
        
        engine.model_manager.model_manager.current_model = mock_model
        engine.model_manager.model_manager.current_model_name = model_name
        return True
    
    import app.domain.training.services
    import app.domain.detection.services
    
    monkeypatch.setattr(app.domain.training.services, "train_model", mock_train_model)
    monkeypatch.setattr(app.domain.detection.services, "predict_image", mock_predict_image)
    monkeypatch.setattr(engine.model_manager.model_manager, "load_model", mock_load_model)

@pytest.fixture
def client():
    # Use FastAPI TestClient
    with TestClient(app.main.app) as c:
        yield c
