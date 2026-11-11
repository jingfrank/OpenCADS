import torch.nn as nn
import torch

class Aggregater(nn.Module):
  def __init__(self,target_dim):
    super(Aggregater, self).__init__()
    self.target_dim = target_dim
    self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_dim)
  def forward(self,features):
    # feat (B*num_patches,C,P,P)
    unified_features = []
    
    for feat in features:
      feat = feat.reshape(len(feat),1,-1) #-->(B*num_patches,1,C*P*P)
      feat = self.adaptive_pool(feat).squeeze(1) #--> (B*num_patches,target_dim)
      unified_features.append(feat)
    unified_features = torch.stack(unified_features,dim=1) # (B*num_patches,layers,target_dim)
    unified_features = unified_features.reshape(len(unified_features),1,-1)
    unified_features = self.adaptive_pool(unified_features).squeeze(1)
    return unified_features #(B*num_patches,target_dim)