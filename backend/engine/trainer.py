import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from .patchcore import PatchCore
from .model_manager import model_manager
import logging

logger = logging.getLogger("ads.trainer")


class ImageFolderDataset(Dataset):
    """Dataset for loading images from a folder (supports nested directories)."""
    
    def __init__(self, folder_path, transform=None):
        self.files = []
        self.transform = transform
        
        # Recursively find all image files
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    self.files.append(os.path.join(root, f))
        
        logger.info(f"Found {len(self.files)} images in {folder_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def get_transform(image_size=256):
    """Get the standard transform for PatchCore."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_model(data_path, model_name="custom_model.pth", 
                backbone_name="wide_resnet50_2",
                layers_to_extract=(2, 3),
                patch_size=3,
                stride=1,
                target_dim=1024,
                coreset_percentage=1,
                n_nearest_neighbour=3,
                batch_size=8,
                image_size=256,
                progress_callback=None):
    """
    Train a PatchCore model on images from data_path.
    
    Args:
        data_path: Path to folder containing training images (normal samples only)
        model_name: Name for the saved model file
        backbone_name: Feature extractor backbone (resnet18, resnet50, etc.)
        layers_to_extract: Which layers to extract features from
        patch_size: Size of patches for feature extraction
        stride: Stride for patch extraction
        target_dim: Target dimension for aggregated features
        coreset_percentage: Percentage of features to keep in coreset
        n_nearest_neighbour: Number of nearest neighbors for anomaly scoring
        batch_size: Batch size for feature extraction
        image_size: Input image size (will be resized to this)
        progress_callback: Optional callback(stage, progress) for tracking
    """
    def report(stage, progress):
        logger.info(f"{stage} ({progress}%)")
        if progress_callback:
            progress_callback(stage, progress)

    report("加载数据集", 5)
    logger.info(f"Starting training on {data_path}...")
    logger.info(f"Configuration: backbone={backbone_name}, coreset={coreset_percentage}, image_size={image_size}")
    
    device = model_manager.get_device()
    logger.info(f"Using device: {device}")
    
    # Define transforms
    transform = get_transform(image_size)
    
    # Create dataset and dataloader
    dataset = ImageFolderDataset(data_path, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_path}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    report("初始化 PatchCore 模型", 10)
    
    # Initialize PatchCore model
    model = PatchCore(
        backbone_name=backbone_name,
        layers_to_extract=layers_to_extract,
        patch_size=patch_size,
        stride=stride,
        target_dim=target_dim,
        precentage=coreset_percentage,
        n_nearest_neighbour=n_nearest_neighbour,
        device=device
    )
    model.to(device)
    
    report("特征提取中", 20)
    
    # Train (build memory bank) with progress callback
    logger.info("Building memory bank...")
    model.fit(dataloader, coreset_sampling_ratio=coreset_percentage, progress_callback=progress_callback)
    
    report("保存模型", 90)
    
    # Save model
    model_manager.save_model(model, model_name)
    
    # Set as current model
    model_manager.current_model = model
    model_manager.current_model_name = model_name
    
    report("训练完成", 100)
    logger.info(f"Training complete. Model saved as {model_name}")
    return model
