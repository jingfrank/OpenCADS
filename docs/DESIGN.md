# 芯片图像异常检测系统 - 软件设计文档

## 1. 系统概述

本系统是一个基于 PatchCore 算法的无监督图像异常检测系统，专为芯片图像质量检测设计。系统采用前后端分离架构，支持 Docker 容器化部署。

### 1.1 主要功能
- **图像数据管理**：上传、管理训练数据集
- **模型训练**：基于正常样本训练异常检测模型
- **实时检测**：对上传图像进行异常检测，生成热力图

### 1.2 技术栈
| 层级 | 技术 |
|------|------|
| 前端 | React + Vite + TailwindCSS |
| 后端 | FastAPI + PyTorch |
| 部署 | Docker + Nginx |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                        │
├─────────────────────┬───────────────────────────────────┤
│    Frontend (Nginx) │          Backend (FastAPI)         │
│    Port: 80         │          Port: 8000                │
├─────────────────────┴───────────────────────────────────┤
│                    Volumes (数据持久化)                   │
│    - datasets/      - models/      - uploads/           │
└─────────────────────────────────────────────────────────┘
```

### 2.1 模块结构

```
backend/
├── main.py                 # FastAPI 主应用
├── engine/
│   ├── patchcore.py        # PatchCore 算法核心
│   ├── feature_extractor.py # 特征提取器 (WideResNet50)
│   ├── patch_extractor.py  # Patch 提取
│   ├── aggregater.py       # 特征聚合
│   ├── sampler.py          # Coreset 采样
│   ├── memorybank.py       # 特征库 (FAISS)
│   ├── trainer.py          # 训练模块
│   ├── inference.py        # 推理模块
│   └── model_manager.py    # 模型管理
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   │   ├── Layout.jsx      # 主布局
│   │   └── Sidebar.jsx     # 侧边导航
│   └── pages/
│       ├── DatasetsPage.jsx   # 数据集管理
│       ├── TrainingPage.jsx   # 模型训练
│       ├── ModelsPage.jsx     # 模型管理
│       └── DetectionPage.jsx  # 实时检测
└── package.json
```

---

## 3. 核心算法 - PatchCore

### 3.1 算法流程

```
训练阶段:
正常图像 → WideResNet50特征提取 → Patch提取 → 特征聚合 → Coreset采样 → Memory Bank

推理阶段:
测试图像 → 特征提取 → 与Memory Bank KNN查询 → 距离计算 → 异常分数/热力图
```

### 3.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| backbone_name | wide_resnet50_2 | 特征提取网络 |
| layers_to_extract | (2, 3) | 提取的网络层 |
| patch_size | 3 | Patch 大小 |
| target_dim | 1024 | 聚合后特征维度 |
| precentage | 0.1 | Coreset 采样比例 |
| n_nearest_neighbour | 1 | KNN 近邻数 |

### 3.3 异常阈值计算

系统自动基于训练数据计算阈值：
```python
threshold = mean(normal_scores) + 3 * std(normal_scores)
```

---

## 4. API 接口设计

### 4.1 数据集管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/datasets | 获取数据集列表 |
| POST | /api/datasets?category=名称 | 上传数据集 |
| DELETE | /api/datasets/{name} | 删除数据集 |

### 4.2 模型管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/models | 获取模型列表 |
| POST | /api/models/load | 加载模型 |
| POST | /api/upload-model | 上传模型文件 |

### 4.3 训练与检测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/train | 启动模型训练 |
| GET | /api/training-status | 获取训练状态 |
| POST | /api/predict | 图像异常检测 |

### 4.4 预测返回格式

```json
{
    "score": 655.1183,
    "heatmap": "base64_encoded_png...",
    "is_anomaly": false,
    "threshold": 755.1044
}
```

---

## 5. 部署架构

### 5.1 Docker 配置

```yaml
services:
  frontend:
    build: ./frontend
    ports: ["80:80"]
    
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes:
      - ./backend:/app
      - ./backend/models:/app/models
      - ./backend/datasets:/app/datasets
```

### 5.2 数据持久化

| 目录 | 说明 |
|------|------|
| backend/models/ | 训练好的模型文件 (.pth) |
| backend/datasets/ | 上传的训练数据集 |
| backend/uploads/ | 临时上传文件 |
| backend/weights/ | 预训练骨干网络权重 |

---

## 6. 扩展性设计

- **骨干网络可替换**：支持 ResNet18/50, WideResNet50 等
- **算法可扩展**：模块化设计，可替换检测算法
- **GPU 支持**：自动检测 CUDA，支持 GPU 加速
