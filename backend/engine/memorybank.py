import torch.nn as nn
import torch
import faiss
import numpy as np


class MemoryBank(nn.Module):
  def __init__(self, n_nearest_neighbour : int,device : str = 'cpu'):
    super(MemoryBank, self).__init__()
    self.n_nearest_neighbour = n_nearest_neighbour
    self.device = device
    self.bank = torch.tensor([], device=device)  # 初始化为空张量
    self.index = None

  def create_index(self,feature_dim):
    
    if self.device == 'cuda':
      res = faiss.StandardGpuResources()
      return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), feature_dim, faiss.GpuIndexFlatConfig()
            )
    return faiss.IndexFlatL2(feature_dim)

  def add(self,features):
    if isinstance(features, np.ndarray):  # 如果是 NumPy 数组，转成 Tensor
      features = torch.from_numpy(features).to(self.device)
    features = features.to(self.device)  # 确保在正确的设备上
    features = features.detach()  # 确保无需梯度
    #更新FAISS索引
    if self.index:
      self.index.reset()
    # 如果是第一次添加特征，根据特征维度创建索引
    if self.index is None:
        self.index = self.create_index(features.shape[-1])
    
    self.bank = torch.cat([self.bank,features],dim = 0)
    
    self.index.add(self.bank.cpu().numpy()) 

  def knn_query(self,query_features,n_nearest_neighbour=None):
    if n_nearest_neighbour is None:
      n_nearest_neighbour = self.n_nearest_neighbour
    if isinstance(query_features, torch.Tensor):  ## 确保 query_features 是 NumPy 数组
      query_features = query_features.detach().cpu().numpy()  # CUDA -> CPU -> NumPy
    distance,indices = self.index.search(query_features,n_nearest_neighbour)
    return distance,indices
  
  def save_state(self):
    index_to_save = self.index
    if self.device == 'cuda':
        # 创建一个CPU版本的索引
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        index_to_save = cpu_index
    return {
        'device':self.device,
        'n_nearest_neighbour':self.n_nearest_neighbour,
        'bank':self.bank,
        'index':faiss.serialize_index(index_to_save),
    }

  @classmethod
  def load_state(cls,state,device = 'cpu'):
    #创建实例
    memory_bank = cls(
        device = device,
        n_nearest_neighbour = state['n_nearest_neighbour'],
    )
    #恢复状态
    memory_bank.bank = state['bank'].to(device)
    memory_bank.index = faiss.deserialize_index(state['index'])
    if device == 'cuda':
      res = faiss.StandardGpuResources()
      memory_bank.index = faiss.index_cpu_to_gpu(res,0,memory_bank.index)
    return memory_bank