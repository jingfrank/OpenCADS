import torch.nn as nn
import torch
import torch.nn.functional as F

class PatchExtractor(nn.Module):
  # Args:[B,C,H,W]
  # output:[B,N,C,P,P]
  def __init__(self, patch_size=3, stride=1):
    super(PatchExtractor, self).__init__()
    self.patch_size = patch_size
    self.patchsize = patch_size  # Alias for backward compatibility
    self.stride = stride
  def forward(self,features,return_spatial_info=False):
    B,C,H,W = features.shape
    P = self.patch_size
    padding = int((self.patch_size - 1) / 2)

    # 使用 unfold 操作提取 Patches
    patches = F.unfold(features, kernel_size=P, stride=self.stride,padding=padding,dilation=1) #[B,C*P*P,N]
    num_patches_h = (H + 2 * padding - (P-1)-1) / self.stride + 1
    num_patches_w = (W + 2 * padding - (P-1)-1) / self.stride + 1
    num_patches = [int(num_patches_h ), int(num_patches_w)]
    #(h*,w*)
    patches = patches.view(B, C, P, P, -1)  # 形状: (B, C, P, P, N)
    patches = patches.permute(0, 4, 1, 2, 3)  # 形状: (B, N, C, P, P)
    
    if return_spatial_info:
      return patches,num_patches
    return patches
