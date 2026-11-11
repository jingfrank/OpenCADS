# 异常检测系统 API 文档

本文档描述了异常检测系统后端 API 的接口规范。

## 基础 URL (Base URL)
API 服务位于后端服务器的根路径。
默认开发环境 URL: `http://localhost:8000`

## 系统接口 (System Endpoints)

### 1. 根路径检查 (Root Check)
*   **路径**: `/`
*   **方法**: `GET`
*   **描述**: 基本连接性检查。
*   **响应**:
    ```json
    {
      "status": "ok",
      "message": "Chip Anomaly Detection System Backend is running"
    }
    ```

### 2. 健康检查 (Health Check)
*   **路径**: `/api/health`
*   **方法**: `GET`
*   **描述**: 检查系统是否运行正常。
*   **响应**:
    ```json
    {
      "status": "ok",
      "message": "System Operational"
    }
    ```

## 模型管理 (Model Management)

### 3. 获取模型列表 (List Models)
*   **路径**: `/api/models`
*   **方法**: `GET`
*   **描述**: 返回可用模型列表以及当前加载的模型名称。
*   **响应**:
    ```json
    {
      "models": ["model1_name", "model2_name"],
      "current_model": "model1_name"
    }
    ```

### 4. 加载模型 (Load Model)
*   **路径**: `/api/models/load`
*   **方法**: `POST`
*   **描述**: 加载指定的模型到内存中以进行推理。
*   **参数**:
    *   `model_name` (查询参数 Query Parameter, 必填): 要加载的模型名称（必须存在于模型列表中）。
*   **响应**:
    ```json
    {
      "status": "success",
      "message": "Loaded model <model_name>"
    }
    ```

### 5. 上传模型 (Upload Model)
*   **路径**: `/api/models/upload`
*   **方法**: `POST`
*   **描述**: 上传预训练的模型文件 (`.pth`)。
*   **请求体**: `multipart/form-data`
    *   `file`: 模型文件 (文件名必须以 `.pth` 结尾)。
*   **响应**:
    ```json
    {
      "status": "success",
      "message": "Uploaded model <filename>"
    }
    ```

## 训练 (Training)

### 6. 训练模型 (Train Model)
*   **路径**: `/api/train`
*   **方法**: `POST`
*   **描述**: 在后台启动训练进程。需上传包含训练图像的 ZIP 文件。
*   **请求体**: `multipart/form-data`
    *   `file`: 包含训练数据集的 ZIP 文件。
*   **响应**:
    ```json
    {
      "status": "success",
      "message": "Training started in background",
      "model_name": "<generated_model_name>.pth"
    }
    ```
*   **注意**: 模型名称将根据 zip 文件名自动生成并进行字符清洗。

## 推理 (Inference)

### 7. 预测 (Predict)
*   **路径**: `/api/predict`
*   **方法**: `POST`
*   **描述**: 对单张图像进行异常检测。**需要先加载模型。**
*   **请求体**: `multipart/form-data`
    *   `file`: 要分析的图像文件。
*   **响应**:
    ```json
    {
      "score": 0.1234,              // 异常分数 (浮点数)
      "heatmap": "base64_string...", // 热力图叠加层的 Base64 编码 PNG 字符串
      "is_anomaly": false,          // 基于阈值的布尔判定结果
      "threshold": 0.5              // 用于判定的阈值
    }
    ```

## 数据集管理 (Dataset Management)

### 8. 获取数据集列表 (List Datasets)
*   **路径**: `/api/datasets`
*   **方法**: `GET`
*   **描述**: 列出所有受管数据集及其元数据（图像数量、大小、缩略图）。
*   **响应**:
    ```json
    {
      "datasets": [
        {
          "name": "dataset_name",
          "image_count": 10,
          "size": "5.2 MB",
          "created_at": "2024-01-01 12:00",
          "thumbnail": "base64_string..."
        }
      ]
    }
    ```

### 9. 创建/更新数据集 (Create/Update Dataset)
*   **路径**: `/api/datasets`
*   **方法**: `POST`
*   **描述**: 创建新数据集或向现有数据集添加图像。支持 ZIP 文件或单张图像。
*   **参数**:
    *   `category` (查询参数 Query Parameter, 必填): 数据集类别名称。
*   **请求体**: `multipart/form-data`
    *   `file`: 图像文件或包含图像的 ZIP 文件。
*   **响应**:
    ```json
    {
      "status": "success",
      "message": "Dataset '<name>' created/updated",
      "dataset": { ... } // 更新后的数据集元数据
    }
    ```

### 10. 删除数据集 (Delete Dataset)
*   **路径**: `/api/datasets/{name}`
*   **方法**: `DELETE`
*   **描述**: 永久删除数据集及其文件。
*   **参数**:
    *   `name` (路径参数 Path Parameter): 要删除的数据集名称。
*   **响应**:
    ```json
    {
      "status": "success",
      "message": "Dataset '<name>' deleted"
    }
    ```
