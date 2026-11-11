import os

MODELS_DIR = "models"

def _get_device():
    """Lazily detect CUDA availability so torch import errors don't break startup."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        self._device = None  # resolved lazily on first use
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)

    @property
    def device(self):
        if self._device is None:
            self._device = _get_device()
        return self._device

    def load_model(self, model_name):
        """Load a PatchCore model from file using the class method."""
        from .patchcore import PatchCore

        path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {model_name} not found")

        model = PatchCore.load(path, device=self.device)

        self.current_model = model
        self.current_model_name = model_name
        print(f"Loaded model: {model_name} on {self.device}")
        return True

    def save_model(self, model, name):
        """Save a PatchCore model to file safely."""
        path = os.path.join(MODELS_DIR, name)
        temp_path = path + ".tmp"

        try:
            model.save(temp_path)
            os.replace(temp_path, path)
            print(f"Saved model safely to {path}")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Failed to save model: {e}")
            raise

    def list_models(self):
        """List all available .pth model files."""
        if not os.path.exists(MODELS_DIR):
            return []
        return [f for f in os.listdir(MODELS_DIR) if f.endswith(".pth")]

    def list_models_detail(self):
        """List models with rich metadata extracted from checkpoint files.
        
        torch is imported lazily inside this method so the rest of the API
        keeps working even when torch is not installed.
        """
        if not os.path.exists(MODELS_DIR):
            return []

        results = []
        for f in os.listdir(MODELS_DIR):
            if not f.endswith(".pth"):
                continue

            path = os.path.join(MODELS_DIR, f)
            stat = os.stat(path)

            info = {
                "name": f,
                "file_size": stat.st_size,
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime,
                "backbone": None,
                "coreset_ratio": None,
                "n_neighbors": None,
                "threshold": None,
                "train_score_mean": None,
                "train_score_std": None,
                "memory_bank_size": None,
                "feature_dim": None,
            }

            # Try to extract metadata from checkpoint without full model instantiation
            try:
                import torch
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)

                if "init_params" in checkpoint:
                    p = checkpoint["init_params"]
                    info["backbone"] = p.get("backbone_name")
                    info["coreset_ratio"] = p.get("precentage")
                    info["n_neighbors"] = p.get("n_nearest_neighbour")
                    info["feature_dim"] = p.get("target_dim")

                info["threshold"] = checkpoint.get("threshold")
                info["train_score_mean"] = checkpoint.get("train_score_mean")
                info["train_score_std"] = checkpoint.get("train_score_std")

                mb_state = checkpoint.get("memory_bank_state")
                if mb_state and "bank" in mb_state:
                    bank = mb_state["bank"]
                    if hasattr(bank, "shape"):
                        info["memory_bank_size"] = bank.shape[0]

                del checkpoint
            except ImportError:
                # torch not available – return file-system metadata only
                pass
            except Exception as e:
                print(f"Warning: could not read metadata from {f}: {e}")

            results.append(info)

        results.sort(key=lambda x: x["modified_at"], reverse=True)
        return results

    def get_model(self):
        """Get the currently loaded model."""
        return self.current_model

    def get_device(self):
        """Get the current device."""
        return self.device


model_manager = ModelManager()
