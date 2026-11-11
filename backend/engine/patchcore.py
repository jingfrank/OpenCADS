import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from sklearn.random_projection import SparseRandomProjection
import faiss
from .patch_extractor import PatchExtractor
from .feature_extractor import FeatureExtractor
from .aggregater import Aggregater
from .sampler import ApproximateGreedyCoresetSampler
from .memorybank import MemoryBank
from .common import RescaleSegmentor
import os as _os

# Default pretrained weights path
_DEFAULT_WEIGHTS_DIR = _os.path.join(_os.path.dirname(__file__), "..", "weights")
_DEFAULT_WEIGHTS_PATH = _os.path.join(_DEFAULT_WEIGHTS_DIR, "wide_resnet50_2_features_only.pth")

class PatchCore(nn.Module):
    def __init__(self, 
                backbone_name="wide_resnet50_2", 
                layers_to_extract=(2,3),
                patch_size=3,
                stride=1,
                target_dim = 1024,
                precentage=0.1,
                n_nearest_neighbour = 1, 
                pretrained_weights_path = None,
                device="cpu"):
        super().__init__()
        self.device = device

        self.forward_modules = nn.ModuleDict()
        # Use default weights path if not specified and file exists
        if pretrained_weights_path is None and _os.path.exists(_DEFAULT_WEIGHTS_PATH):
            pretrained_weights_path = _DEFAULT_WEIGHTS_PATH
        
        self.forward_modules["feature_extractor"] = FeatureExtractor(backbone_name=backbone_name, layers_to_extract=layers_to_extract, device=device, pretrained_weights_path=pretrained_weights_path)
        self.forward_modules["patch_extractor"] = PatchExtractor(patch_size=patch_size, stride=stride)
        self.forward_modules["aggregater"] = Aggregater(target_dim=target_dim)
        # coreset_sampler is not a nn.Module, store as regular attribute
        self.coreset_sampler = ApproximateGreedyCoresetSampler(precentage=precentage)
        self.memory_bank = MemoryBank(n_nearest_neighbour=n_nearest_neighbour,device=device)
        self.rescale_segmentor = RescaleSegmentor()
        
        # Threshold for anomaly detection (computed during training)
        self.threshold = None
        self.threshold_std_multiplier = 3.0  # mean + N*std
    def _embed(self, images,provide_patch_shapes = False):
        with torch.no_grad():
            features = self.forward_modules["feature_extractor"](images.to(self.device))
        
        # Resize features to match the largest spatial resolution (layer 2)
        # Layer 2: (B, C2, H2, W2), Layer 3: (B, C3, H3, W3)
        # We upsample Layer 3 to H2, W2
        patch_outputs = [self.forward_modules["patch_extractor"](feature,return_spatial_info=True) for feature in features]
        patches = [patch_output[0] for patch_output in patch_outputs]

        num_patches = [patch_output[1] for patch_output in patch_outputs]

        ref_num_patch = num_patches[0]

        for i in range(1, len(patches)):
            patch = patches[i]
            num_patch = num_patches[i]
            B, N, C, P, P = patch.shape
            #将维度转换为(B,num_patch_h,num_patch_w,C,P,P)
            patch = patch.view(B,num_patch[0],num_patch[1],C,P,P)
            #交换维度为 (B, C, P, P, num_patches_h, num_patches_w)
            patch = patch.permute(0, -3, -2, -1, 1, 2)
            #记录当前形状
            patch_base_shape = patch.shape
            patch = patch.reshape(-1, *patch.shape[-2:])
            patch = F.interpolate(patch.unsqueeze(1),
                                    size =(ref_num_patch[0],ref_num_patch[1]),
                                    mode ="bilinear",
                                    align_corners = False,)
            patch = patch.squeeze(1)
            #还原为(B, C, P, P, ref_num_patches_h, ref_num_patches_w)
            patch = patch.reshape(*patch_base_shape[:-2],ref_num_patch[0],ref_num_patch[1])

            patch = patch.permute(0, -2, -1, 1, 2, 3) #(B,ref_num_patches_h,ref_num_patches_w,C,P,P)
            patch = patch.reshape(len(patch), -1, *patch.shape[-3:]) #(B,ref_num_patches_h*ref_num_patches_w,C,P,P)
            patches[i] = patch #(B, num_patches, C, P, P)
        
        patches = [x.reshape(-1, *x.shape[-3:]) for x in patches]
        features = self.forward_modules["aggregater"](patches) #[B,C]

        if provide_patch_shapes:
            return features, num_patches
        return features

    def fit(self, train_loader, coreset_sampling_ratio=0.1, progress_callback=None):
        """
        Build memory bank from training images.
        """
        def report(stage, progress):
            if progress_callback:
                progress_callback(stage, progress)

        print("Extracting features...")
        embeddings = []
        total_batches = len(train_loader)
        for batch_idx, images in enumerate(train_loader):
            emb = self._embed(images)
            embeddings.append(emb)
            # Report progress: feature extraction spans 20% - 60%
            pct = 20 + int((batch_idx + 1) / total_batches * 40)
            report(f"特征提取中 ({batch_idx + 1}/{total_batches})", pct)
        
        embeddings = np.concatenate(embeddings, axis=0)

        report("核心集采样中", 65)
        features = self.coreset_sampler.run(embeddings)
        self.memory_bank.add(features)
        
        # Compute threshold based on training set scores
        report("计算异常阈值", 75)
        print("Computing anomaly threshold from training data...")
        self._compute_threshold(train_loader)
        print(f"Threshold set to: {self.threshold:.4f}")
        report("阈值计算完成", 85)

        return
    
    def _compute_threshold(self, train_loader):
        """Compute anomaly threshold based on training set score distribution."""
        scores = []
        for images in train_loader:
            image_scores, _ = self.predict(images)
            if isinstance(image_scores, np.ndarray):
                scores.extend(image_scores.flatten().tolist())
            else:
                scores.append(float(image_scores))
        
        scores = np.array(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Threshold = mean + N * std (default N=3)
        self.threshold = mean_score + self.threshold_std_multiplier * std_score
        self.train_score_mean = mean_score
        self.train_score_std = std_score
        

    def predict(self, image_tensor):
        """
        Predict anomaly score and map for a single image batch.
        """
        with torch.no_grad():
            embedding, num_patches = self._embed(image_tensor, provide_patch_shapes=True)
        
        b = image_tensor.shape[0]
        
        # num_patches is a list of [h, w] for each layer, use first layer's shape
        ref_num_patches = num_patches[0]  # [h, w]

        distance , indices = self.memory_bank.knn_query(embedding)  #distance [b*num_patches,k]
        distance = np.mean(distance,axis = -1)
        anomaly_map = distance.reshape(b, ref_num_patches[0], ref_num_patches[1])
        anomaly_map = self.rescale_segmentor.convert_to_segmentation(anomaly_map)
        image_scores = np.max(anomaly_map,axis = -1)
        
        return image_scores, anomaly_map


    def save(self, path):
        """Save the PatchCore model to a file."""
        # Collect init parameters for reconstruction
        init_params = {
            'backbone_name': self.forward_modules["feature_extractor"].backbone_name,
            'layers_to_extract': self.forward_modules["feature_extractor"].layers_to_extract,
            'patch_size': self.forward_modules["patch_extractor"].patch_size,
            'stride': self.forward_modules["patch_extractor"].stride,
            'target_dim': self.forward_modules["aggregater"].target_dim,
            'precentage': self.coreset_sampler.precentage,
            'n_nearest_neighbour': self.memory_bank.n_nearest_neighbour,
            'device': self.device,
        }
        
        checkpoint = {
            'init_params': init_params,
            'forward_modules_state': self.forward_modules.state_dict(),
            'memory_bank_state': self.memory_bank.save_state(),
            'threshold': self.threshold,
            'train_score_mean': getattr(self, 'train_score_mean', None),
            'train_score_std': getattr(self, 'train_score_std', None),
        }
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, device='cpu'):
        """Load a PatchCore model from a file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Reconstruct the model with saved parameters
        init_params = checkpoint['init_params']
        init_params['device'] = device  # Override device
        
        model = cls(**init_params)
        
        # Load forward_modules state
        model.forward_modules.load_state_dict(checkpoint['forward_modules_state'])
        
        # Load memory_bank state
        model.memory_bank = MemoryBank.load_state(checkpoint['memory_bank_state'], device=device)
        
        # Load threshold and score statistics
        model.threshold = checkpoint.get('threshold', None)
        model.train_score_mean = checkpoint.get('train_score_mean', None)
        model.train_score_std = checkpoint.get('train_score_std', None)
        
        model.to(device)
        print(f"Model loaded from {path}")
        if model.threshold is not None:
            print(f"Threshold: {model.threshold:.4f}")
        
        return model
