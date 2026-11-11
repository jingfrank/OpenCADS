import torch
import torch.nn as nn
import numpy as np
import tqdm

class GreedySampler:
    def __init__(self,percent: float,device :torch.device,projection_dim:int):
        self.percent = percent
        self.device = device
        self.projection_dim = projection_dim

    def _project(self,features):
        """
        radomly linear projection
        features:[B*N,C]
        """
        # 计算投影
        mapper = torch.nn.Linear(features.shape[1],self.projection_dim,bias=False).to(self.device)
        features = features.to(self.device) 
        projection = mapper(features)
        return projection

    def run(self,features):
        """
        运行Greedy采样
        :param features: 特征矩阵，形状为 (B*N, C)
        :return: 采样后的特征矩阵，形状为 (B*N, C)
        """
        if isinstance(features,np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._project(features)
        sample_indices = self._sample_greedy(reduced_features)
        featurs= features[sample_indices]

        return featurs

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _sample_greedy(self,features):
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percent)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler():
    def __init__(self, precentage: float = 0.1, device='cpu', number_of_starting_points: int = 10, dimension_to_project_features_to: int = 128):
        self.number_of_starting_points = number_of_starting_points
        self.precentage = precentage
        self.percent = precentage  # Alias
        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _project(self,features):
        """
        radomly linear projection
        features:[B*N,C]
        """
        # 计算投影
        mapper = torch.nn.Linear(features.shape[1],self.dimension_to_project_features_to,bias=False).to(self.device)
        features = features.to(self.device) 
        projection = mapper(features)
        return projection

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()
    
    def run(self,features):
        """
        运行Greedy采样
        :param features: 特征矩阵，形状为 (B*N, C)
        :return: 采样后的特征矩阵，形状为 (B*N, C)
        """
        if isinstance(features,np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._project(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        featurs= features[sample_indices]

        return featurs

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percent)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)
