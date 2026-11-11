import torch
from torchvision import transforms
from PIL import Image
from .model_manager import model_manager
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt


def get_transform(image_size=256):
    """Get the standard transform for PatchCore inference."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_image(image_file, threshold=0.5):
    """
    Predict anomaly score and generate heatmap for an image.
    
    Args:
        image_file: File-like object or path to image
        threshold: Score threshold for anomaly classification (not used in scoring, just for reference)
    
    Returns:
        dict with 'score' (float) and 'heatmap' (base64 encoded PNG)
    """
    model = model_manager.get_model()
    if model is None:
        raise ValueError("No model loaded. Please load or train a model first.")
    
    # Load and preprocess image
    image = Image.open(image_file).convert("RGB")
    original_size = image.size  # (width, height)
    original_image = image.copy()  # Keep original for overlay
    
    transform = get_transform(256)
    input_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    
    # Move to same device as model
    device = model_manager.get_device()
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    scores, anomaly_maps = model.predict(input_tensor)
    
    # scores shape: (batch_size,) or scalar
    # anomaly_maps shape: (batch_size, H, W) or (H, W)
    
    # Handle different output shapes
    # scores may be array or scalar
    if isinstance(scores, (list, np.ndarray)):
        scores = np.array(scores)
        score = float(scores.flatten()[0])
    else:
        score = float(scores)
    
    # anomaly_maps may be list of arrays from RescaleSegmentor
    if isinstance(anomaly_maps, list):
        anomaly_map = np.array(anomaly_maps[0])  # Take first batch
    elif isinstance(anomaly_maps, np.ndarray):
        if anomaly_maps.ndim == 3:
            anomaly_map = anomaly_maps[0]  # Take first batch
        else:
            anomaly_map = anomaly_maps
    else:
        anomaly_map = np.array(anomaly_maps)
    
    # Normalize anomaly map to 0-1 for better visualization
    if anomaly_map.max() > anomaly_map.min():
        anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    else:
        anomaly_map_normalized = anomaly_map
    
    # Create heatmap overlay visualization
    heatmap_base64 = create_heatmap_overlay(original_image, anomaly_map_normalized)
    
    # Create binary mask (defect localization)
    mask_base64 = create_binary_mask(anomaly_map_normalized)
    
    # Use model's threshold if available, otherwise use the provided threshold
    model_threshold = getattr(model, 'threshold', None)
    effective_threshold = model_threshold if model_threshold is not None else threshold
    
    return {
        "score": float(score),
        "heatmap": heatmap_base64,
        "mask": mask_base64,
        "is_anomaly": bool(score > effective_threshold),
        "threshold": float(effective_threshold)
    }


def create_heatmap_overlay(original_image, anomaly_map, colormap='jet', alpha=0.5):
    """
    Create a heatmap overlay on the original image.
    
    Args:
        original_image: PIL Image object
        anomaly_map: 2D numpy array of normalized anomaly scores (0-1)
        colormap: Matplotlib colormap name
        alpha: Transparency of heatmap overlay (0-1)
    
    Returns:
        Base64 encoded PNG image string
    """
    # Resize original image to match anomaly map size or vice versa
    target_size = (256, 256)
    original_resized = original_image.resize(target_size, Image.Resampling.LANCZOS)
    original_array = np.array(original_resized)
    
    # Resize anomaly map to target size
    from scipy.ndimage import zoom
    h, w = anomaly_map.shape
    if h != target_size[1] or w != target_size[0]:
        zoom_factors = (target_size[1] / h, target_size[0] / w)
        anomaly_map_resized = zoom(anomaly_map, zoom_factors, order=1)
    else:
        anomaly_map_resized = anomaly_map
    
    # Apply colormap to anomaly map
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colored = cmap(anomaly_map_resized)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend original image with heatmap
    original_float = original_array.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    blended = (1 - alpha) * original_float + alpha * heatmap_float
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image and then to base64
    blended_image = Image.fromarray(blended)
    
    buf = BytesIO()
    blended_image.save(buf, format='PNG')
    buf.seek(0)
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return heatmap_base64


def create_binary_mask(anomaly_map, threshold=0.5, target_size=(256, 256)):
    """
    Create a binary mask from normalized anomaly map.
    
    Args:
        anomaly_map: 2D numpy array of normalized anomaly scores (0-1)
        threshold: Threshold for binarization (0-1)
        target_size: Output image size (width, height)
    
    Returns:
        Base64 encoded PNG image string (white=anomaly, black=normal)
    """
    from scipy.ndimage import zoom
    
    h, w = anomaly_map.shape
    if h != target_size[1] or w != target_size[0]:
        zoom_factors = (target_size[1] / h, target_size[0] / w)
        anomaly_map_resized = zoom(anomaly_map, zoom_factors, order=1)
    else:
        anomaly_map_resized = anomaly_map
    
    # Binarize: above threshold = white (255), below = black (0)
    binary_mask = (anomaly_map_resized > threshold).astype(np.uint8) * 255
    
    # Convert to PIL Image and then to base64
    mask_image = Image.fromarray(binary_mask, mode='L')
    
    buf = BytesIO()
    mask_image.save(buf, format='PNG')
    buf.seek(0)
    mask_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return mask_base64


def create_heatmap(anomaly_map, original_size=None, colormap='jet'):
    """
    Create a standalone heatmap visualization from anomaly map (legacy function).
    
    Args:
        anomaly_map: 2D numpy array of anomaly scores
        original_size: Optional tuple (width, height) to resize heatmap
        colormap: Matplotlib colormap name
    
    Returns:
        Base64 encoded PNG image string
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    
    # Display heatmap
    im = ax.imshow(anomaly_map, cmap=colormap, interpolation='bilinear')
    ax.axis('off')
    
    # Remove margins
    plt.tight_layout(pad=0)
    
    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return heatmap_base64


def predict_batch(image_files, threshold=0.5):
    """
    Predict anomaly scores for multiple images.
    
    Args:
        image_files: List of file-like objects or paths
        threshold: Score threshold for anomaly classification
    
    Returns:
        List of prediction results
    """
    results = []
    for image_file in image_files:
        try:
            result = predict_image(image_file, threshold)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "score": None,
                "heatmap": None
            })
    return results
