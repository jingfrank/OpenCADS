# OpenCADS: 芯片工业无监督异常检测系统

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/jingfrank/OpenCADS?style=social)](https://github.com/jingfrank/OpenCADS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

[**English**](./README.md) | [**架构设计文档**](./docs/DESIGN.md) | [**接口规范文档**](./docs/API_DOCS.md) | [**性能测试基准**](./docs/BENCHMARK.md)

</div>

---

## 📌 项目概述

**OpenCADS (Chip Anomaly Detection System)** 是一套面向半导体芯片制程与高精工业制造（如晶圆检测、PCB瑕疵排查）的**端到端全栈工业级无监督图像异常检测系统**。

在实际工业产线中，产品良品率通常超过 99.9%，缺陷样本（负样本）极其罕见且收集代价高昂，传统的监督学习分类算法难以落地。OpenCADS 采用先进的 **PatchCore** 无监督自学习算法框架，**仅需无缺陷的正常良品样本即可完成模型构建**。系统可对未知缺陷进行高精度异常评分判定，并自动生成像素级的缺陷定位热力图（Heatmap）与二值掩膜（Binary Mask）。

---

## ✨ 核心特性

- **工业级无监督异常检测 (PatchCore)**
  - 基于预训练 WideResNet-50 骨干网络提取中高层多尺度感受野特征（Layer 2 & 3）。
  - 生成融合局部上下文信息的 1024 维 Patch 局部表征，精准刻画微小缺陷。
- **极致内存压缩与检索提速**
  - **核心集抽样 (Coreset Subsampling)**：在保留全局特征分布包络的前提下，压缩掉 90% 的冗余特征点。
  - **FAISS 向量检索库**：采用 Facebook FAISS 优化 K-NN 最近邻距离搜索，实现单图毫秒级异常评分匹配。
- **自适应动态异常阈值**
  - 在训练阶段根据良品样本得分正态分布自动校准阈值：$\text{Threshold} = \mu_{\text{normal}} + 3\sigma_{\text{normal}}$，无需人工手动反复试凑。
- **现代化工业全栈控制台 (React 18 + FastAPI)**
  - **数据集管理中心**：支持 ZIP 压缩包批量上传、样本缩略图预览与标注状态标记。
  - **异步训练流**：后台非阻塞训练，具备阶段状态追踪与实时进度轮询。
  - **模型热切换**：毫秒级多模型在线加载与切换，支持模型元信息查询。
  - **实时缺陷检测工作室**：原生图、二值掩膜定位图、Jet 伪彩色热力图叠加对照展示。
- **云原生容器化与开箱即用**
  - **极简单机体验**：Docker Compose 预置内嵌 SQLite，一键拉起即用，免去外部数据库配置成本。
  - **企业级分布式拓展**：具备 PostgreSQL + Redis + Celery 架构拓展支持，轻松支撑产线批次高吞吐。

---

## 📊 硬件性能评测基准 (Benchmark)

在单张图像端到端推理评测下（输入尺寸 256×256，WideResNet-50 骨干网络，10% Coreset 抽样比）：

| 硬件计算平台 | 模型推理耗时 (ms) | 端到端总处理耗时 (ms) | 相对 CPU 加速比 (推理) |
| :--- | :---: | :---: | :---: |
| **边缘 CPU (x86)** | 801.53 ms | 857.28 ms | **1.0x (基准)** |
| **NVIDIA Tesla T4** | 151.70 ms | 241.07 ms | **5.3x** |
| **NVIDIA Tesla V100** | **29.71 ms** | **157.38 ms** | **27.0x** |

> *注：端到端总耗时包含：数据预处理、模型前向推理、FAISS 距离比对、高斯滤波平滑与 base64 热图合成全流程。*

---

## 🏗️ 系统架构图

```
┌────────────────────────────────────────────────────────────────────────┐
│                        现代化工业质检前端 (React 18)                   │
│         [数据集管理]         [模型构建工作台]         [实时检测工作室] │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST API (Port 3000 -> 8000)
┌───────────────────────────────────▼────────────────────────────────────┐
│                       FastAPI 应用服务网关                             │
│      ├── /api/datasets        ├── /api/models                          │
│      ├── /api/train           └── /api/predict (缺陷推理服务)          │
└───────────────┬───────────────────────────────────┬────────────────────┘
                │                                   │
┌───────────────▼────────────────┐ ┌────────────────▼────────────────────┐
│      SQLModel 持久化层         │ │      PatchCore 核心算法引擎         │
│  SQLite (开发) / Postgres(生产)│ │  - WideResNet-50 特征抽取器         │
│  - 训练任务状态机              │ │  - Patch 局部表征提取与降维聚合     │
│  - 质检历史回溯与标注数据      │ │  - Coreset 核心集特征压缩采样       │
│                                │ │  - FAISS 高效高维向量 Memory Bank   │
└────────────────────────────────┘ └─────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 推荐：通过 Docker Compose 一键启动

确保本地已安装 [Docker](https://www.docker.com/) 与 [Docker Compose](https://docs.docker.com/compose/)。

```bash
# 克隆仓库
git clone https://github.com/jingfrank/OpenCADS.git
cd OpenCADS

# 一键构建并启动所有容器服务
docker-compose up -d --build
```

启动完成后直接在浏览器中访问：
- **前端质检控制台 UI**：[http://localhost:3000](http://localhost:3000)
- **后端 Swagger 交互式接口文档**：[http://localhost:8000/docs](http://localhost:8000/docs)

停止容器：
```bash
docker-compose down
```

---

### 2. 本地开发环境手动运行

#### 后端服务
```bash
cd backend
python -m venv venv

# Linux / macOS:
source venv/bin/activate
# Windows PowerShell:
venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 前端服务
```bash
cd frontend
npm install
npm run dev
# 前端 Vite 运行在 http://localhost:3000
```

---

## 🧪 自动化测试

运行后端单元测试与端到端接口验证：

```bash
cd backend
pytest tests/ -v
```

---

## 📂 仓库目录结构

```
OpenCADS/
├── backend/
│   ├── app/                      # DDD 领域驱动设计架构
│   │   ├── core/                 # 数据库连接、Celery 任务调度、配置
│   │   └── domain/               # 数据集域、模型域、训练域、质检域
│   ├── engine/                   # PatchCore 算法核心
│   │   ├── feature_extractor.py  # WideResNet 特征抽取
│   │   ├── sampler.py            # Coreset 采样器
│   │   ├── memorybank.py         # FAISS 向量库封装
│   │   ├── trainer.py            # 离线自学习训练与阈值标定
│   │   └── inference.py          # 在线实时推理与热力图合成
│   ├── datasets/                 # 训练与质检数据集目录
│   ├── tests/                    # Pytest 自动化测试套件
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/           # UI 布局与复用组件
│   │   └── pages/                # 数据集页、训练页、模型管理页、检测页
│   ├── package.json
│   └── Dockerfile
├── docs/                         # 技术文档与报告
│   ├── DESIGN.md                 # 系统详细设计文档
│   ├── USER_GUIDE.md             # 用户操作指南手册
│   ├── API_DOCS.md               # RESTful API 规范
│   └── BENCHMARK.md              # 软硬件推理性能基准测试报告
├── docker-compose.yml            # 默认轻量单机启动编排
├── docker-compose.prod.yml       # 分布式生产启动编排 (Postgres+Redis+Celery)
├── LICENSE                       # MIT 开源协议
├── README.md                     # 英文自述
└── README_CN.md                  # 中文自述
```

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 协议开源。
