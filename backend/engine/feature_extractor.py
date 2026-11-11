import torch.nn as nn
import torch
import os
import timm

class FeatureExtractor(nn.Module):

  def __init__(self,backbone_name ="wide_resnet50_2",layers_to_extract=(2,3),device = None,pretrained_weights_path=None,):
    super(FeatureExtractor, self).__init__()
    self.backbone_name = backbone_name
    self.layers_to_extract = layers_to_extract
    self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #加载timm预训练模型
    # 加载本地权重（如果提供了路径）
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
      print(f"Loading backbone weights from local file: {pretrained_weights_path}")
      self.backbone = timm.create_model(
        backbone_name,
        pretrained=False,  # 禁用自动下载
        features_only=True,
      )
      state_dict = torch.load(pretrained_weights_path, map_location=self.device)
      self.backbone.load_state_dict(state_dict)
    else:
      # 回退到在线下载（如果本地权重不存在）
      print("Loading backbone weights from online source...")
      self.backbone = timm.create_model(
        backbone_name,
        pretrained=True,  # 尝试在线下载
        features_only=True,
      )
    self.layers_to_extract = layers_to_extract
    # 设置设备
    
    self.to(self.device)
  def forward(self,x):
    with torch.no_grad():
      self.backbone.eval()
      features = self.backbone(x)
      selected_featrues =[features[i] for i in self.layers_to_extract]
    return selected_featrues

  def feature_dimensions(self,input_shape):
    # 创建模拟输入张量
    _input = torch.ones([1] + list(input_shape)).to(self.device)
    # 前向传播获取各层输出
    _output = self(_input)
    # 提取指定层的特征维度
    return [_output[layer].shape[1] for layer in self.layers_to_extract]