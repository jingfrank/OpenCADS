import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.ndimage as ndimage
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from typing import Tuple

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )


def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )
    
class RescaleSegmentor():
    def __init__(self, device = 'cpu', target_size=256):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]

def compute_batch_mahalanobis_distance_numpy(feature_vector, feature_mean, feature_covinv):
    """
    使用NumPy计算批量马哈拉诺比斯距离
    Args:
        feature_vector: [B, C] 批量特征向量
        feature_mean: [C] 特征均值
        feature_covinv: [C, C] 协方差矩阵的逆
    Returns:
        distances: [B] 每个样本的马哈拉诺比斯距离
    """
    # 中心化特征向量 [B, C]
    center_feature_vector = feature_vector - feature_mean
    
    # 计算批量距离
    # 方法1: 使用einsum (推荐，效率高)
    mahalanobis_squared = np.einsum('bi,ij,bj->b', center_feature_vector, feature_covinv, center_feature_vector)
    
    # 方法2: 使用矩阵乘法
    # temp = np.dot(center_feature_vector, feature_covinv)  # [B, C]
    # mahalanobis_squared = np.sum(temp * center_feature_vector, axis=1)  # [B]
    
    # 确保非负并开平方根
    mahalanobis_distance = np.sqrt(np.maximum(0, mahalanobis_squared))
    
    return mahalanobis_distance

def compute_batch_mahalanobis_distance_pytorch_matmul(feature_vector, feature_mean, feature_covinv, device='cuda'):
    """
    使用PyTorch矩阵乘法版本 (通常更快)
    """
    # 处理 feature_vector (支持 list 类型)
    if isinstance(feature_vector, list):
        if len(feature_vector) > 0:
            if isinstance(feature_vector[0], torch.Tensor):
                feature_vector = torch.stack(feature_vector)
            elif isinstance(feature_vector[0], np.ndarray):
                feature_vector = np.stack(feature_vector)
                feature_vector = torch.from_numpy(feature_vector).float()
            else:
                feature_vector = torch.tensor(feature_vector, dtype=torch.float32)
        else:
            raise ValueError("feature_vector 是空的 list")
    elif isinstance(feature_vector, np.ndarray):
        feature_vector = torch.from_numpy(feature_vector).float()
    elif isinstance(feature_vector, torch.Tensor):
        feature_vector = feature_vector.float()

    # 转换为tensor并移到GPU
    if isinstance(feature_vector, np.ndarray):
        feature_vector = torch.from_numpy(feature_vector).float().to(device)
    if isinstance(feature_mean, np.ndarray):
        feature_mean = torch.from_numpy(feature_mean).float().to(device)
    if isinstance(feature_covinv, np.ndarray):
        feature_covinv = torch.from_numpy(feature_covinv).float().to(device)
    
    feature_vector = feature_vector.to(device)
    # 中心化
    center_feature_vector = feature_vector - feature_mean
    
    # 矩阵乘法: (B, C) @ (C, C) = (B, C)
    temp = torch.mm(center_feature_vector, feature_covinv)
    
    # 逐元素乘法后按行求和: (B, C) * (B, C) -> (B,)
    mahalanobis_squared = torch.sum(temp * center_feature_vector, dim=1)
    
    # 开平方根
    mahalanobis_distance = torch.sqrt(torch.clamp(mahalanobis_squared, min=0))
    
    return mahalanobis_distance.cpu().numpy()  # 返回numpy数组

class PyTorchCovarianceEstimators:
    @staticmethod
    def empirical_covariance(X: torch.Tensor, bias: bool = False) -> torch.Tensor:
        """经验协方差估计"""
        n, p = X.shape
        X_centered = X - X.mean(dim=0, keepdim=True)
        factor = n if bias else n - 1
        return torch.mm(X_centered.t(), X_centered) / factor
    
    @staticmethod
    def shrunk_covariance(X: torch.Tensor, shrinkage: float = 0.1) -> torch.Tensor:
        """收缩协方差估计"""
        emp_cov = PyTorchCovarianceEstimators.empirical_covariance(X)
        p = X.shape[1]
        identity = torch.eye(p, device=X.device, dtype=X.dtype)
        trace = torch.trace(emp_cov)
        
        # 收缩目标：scaled identity matrix
        target = (trace / p) * identity
        
        # 线性收缩
        shrunk_cov = (1 - shrinkage) * emp_cov + shrinkage * target
        return shrunk_cov
    
    @staticmethod
    def ledoit_wolf_covariance(X: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Ledoit-Wolf收缩协方差估计"""
        n, p = X.shape
        X_centered = X - X.mean(dim=0, keepdim=True)

        # 经验协方差
        emp_cov = torch.mm(X_centered.t(), X_centered) / (n - 1)

        # 收缩目标
        trace = torch.trace(emp_cov)
        target = (trace / p) * torch.eye(p, device=X.device, dtype=X.dtype)

        # 估计最优收缩参数
        # 简化版本的Ledoit-Wolf公式
        emp_cov_centered = emp_cov - target
        beta = torch.sum(emp_cov_centered ** 2)

        # 计算delta (这里使用简化版本)
        X_squared = X_centered ** 2
        sample_cov_squared = torch.mm(X_squared.t(), X_squared) / n
        delta = torch.sum(sample_cov_squared - emp_cov ** 2)

        # 最优收缩参数
        shrinkage = torch.clamp(delta / beta, 0, 1).item() if beta > 0 else 0.0

        # 收缩协方差
        shrunk_cov = (1 - shrinkage) * emp_cov + shrinkage * target

        return shrunk_cov, shrinkage

def gaussian_estimation(embeddings: torch.Tensor, 
                       covariance_method: str,
                       shrinkage: float = 0.1,
                       backend: str = "pytorch"):
    """
    计算特征的均值和逆协方差矩阵（支持PyTorch和scikit-learn后端）
    
    Args:
        embeddings: [B, C] 特征向量
        covariance_method: 
            - "EmpiricalCovariance"
            - "ShrunkCovariance" 
            - "LedoitWolf"
        shrinkage: 收缩系数（仅ShrunkCovariance有效）
        backend: "pytorch" 或 "sklearn"
    
    Returns:
        mean: [C] 特征均值
        covinv: [C, C] 协方差矩阵的逆
    """
    # 计算均值（始终使用PyTorch）
    mean = torch.mean(embeddings, dim=0)  # [C]
    
    if backend == "pytorch":
        # PyTorch实现
        if covariance_method == "EmpiricalCovariance":
            cov = PyTorchCovarianceEstimators.empirical_covariance(embeddings)
        elif covariance_method == "ShrunkCovariance":
            cov = PyTorchCovarianceEstimators.shrunk_covariance(embeddings, shrinkage=shrinkage)
        elif covariance_method == "LedoitWolf":
            cov, optimal_shrinkage = PyTorchCovarianceEstimators.ledoit_wolf_covariance(embeddings)
            print(f"Ledoit-Wolf optimal shrinkage: {optimal_shrinkage:.4f}")
        else:
            raise ValueError(f"Unsupported covariance method: {covariance_method}")
            
    elif backend == "sklearn":
        # scikit-learn实现
        X_np = embeddings.cpu().numpy() if embeddings.is_cuda else embeddings.numpy()
        
        if covariance_method == "EmpiricalCovariance":
            estimator = EmpiricalCovariance().fit(X_np)
        elif covariance_method == "ShrunkCovariance":
            estimator = ShrunkCovariance(shrinkage=shrinkage).fit(X_np)
        elif covariance_method == "LedoitWolf":
            estimator = LedoitWolf().fit(X_np)
            print(f"Ledoit-Wolf optimal shrinkage: {estimator.shrinkage_:.4f}")
        else:
            raise ValueError(f"Unsupported covariance method: {covariance_method}")
        
        cov = torch.from_numpy(estimator.covariance_).to(device = embeddings.device, dtype=embeddings.dtype)  # [C, C]
    else:
        raise ValueError(f"Invalid backend: {backend}")

    # 计算逆协方差矩阵（添加小正则项保证数值稳定性）
    cov += torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * 1e-6  # 添加小正则项
    cov_inv = torch.linalg.inv(cov)

    return mean, cov_inv


def compute_mahalanobis_distance_pytorch(feature_vector, feature_mean, feature_covinv, device='cuda'):
    """
    使用PyTorch矩阵乘法版本 (通常更快)
    """
    feature_vector = feature_vector
    # 中心化
    

    center_feature_vector = feature_vector - feature_mean
    # 在计算前统一类型
    center_feature_vector = center_feature_vector
    feature_covinv = feature_covinv
    # 矩阵乘法: (B, C) @ (C, C) = (B, C)
    # temp = torch.mm(center_feature_vector, feature_covinv)
    
    # # 逐元素乘法后按行求和: (B, C) * (B, C) -> (B,)
    # mahalanobis_squared = torch.sum(temp * center_feature_vector, dim=1)
    mahalanobis_squared = torch.einsum('bi,ij,bj->b', center_feature_vector, feature_covinv, center_feature_vector)

    # 开平方根
    mahalanobis_distance = torch.sqrt(torch.clamp(mahalanobis_squared, min=0))
    
    
    return mahalanobis_distance # 返回numpy数组