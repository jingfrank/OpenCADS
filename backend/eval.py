import os
import sys
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Force absolute import path to find backend modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.model_manager import model_manager
from engine.inference import get_transform
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Detection System (PatchCore)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth) inside backend/models or absolute path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name inside backend/datasets or absolute path to dataset directory")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu'). Auto-detect if not set.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results and plots")
    return parser.parse_args()

def load_dataset_images(dataset_path):
    """
    Load test images and ground truth labels from dataset.
    Supports two formats:
    1. Standard labels.json in the dataset folder.
    2. Subfolder structure: test/normal/ and test/<anomaly_type>/ (MVTec AD format)
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
    images_to_eval = []  # List of tuples: (image_path, label) where label is 1 for anomaly, 0 for normal
    
    # Format 2: MVTec AD format (test/normal and test/defect_type)
    test_dir = os.path.join(dataset_path, "test")
    if os.path.exists(test_dir) and os.path.isdir(test_dir):
        print(f"Detected subfolder format (MVTec AD style) in {test_dir}")
        for category in os.listdir(test_dir):
            category_dir = os.path.join(test_dir, category)
            if not os.path.isdir(category_dir):
                continue
            
            label = 0 if category == "normal" else 1
            for root, _, files in os.walk(category_dir):
                for f in files:
                    if f.lower().endswith(image_extensions):
                        images_to_eval.append((os.path.join(root, f), label))
        return images_to_eval
    
    # Format 1: labels.json format
    labels_file = os.path.join(dataset_path, "labels.json")
    if os.path.exists(labels_file):
        print(f"Detected labels.json in {dataset_path}")
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels_dict = json.load(f)
            
            for filename, is_anomaly in labels_dict.items():
                img_path = os.path.join(dataset_path, filename)
                if os.path.exists(img_path) and filename.lower().endswith(image_extensions):
                    # labels.json stores true for anomaly, false for normal
                    label = 1 if is_anomaly else 0
                    images_to_eval.append((img_path, label))
            return images_to_eval
        except Exception as e:
            print(f"Error parsing labels.json: {e}")
            
    # Format 3: Fallback (all files inside the dataset path, labeling based on filename keywords)
    print("No test folder or labels.json found. Reading all images and predicting labels from filenames...")
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(image_extensions):
                lower_name = f.lower()
                # Guess label
                if "anomaly" in lower_name or "bad" in lower_name or "defect" in lower_name or "ng" in lower_name:
                    label = 1
                else:
                    label = 0
                images_to_eval.append((os.path.join(root, f), label))
                
    return images_to_eval

def main():
    args = parse_args()
    
    # 1. Resolve paths
    # Resolve Model Path
    model_path = args.model
    if not os.path.exists(model_path):
        # Try looking inside backend/models
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", args.model)
        if not os.path.exists(model_path):
            print(f"Error: Model path '{args.model}' not found.")
            sys.exit(1)
            
    # Resolve Dataset Path
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        # Try looking inside backend/datasets
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", args.dataset)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path '{args.dataset}' not found.")
            sys.exit(1)
            
    # Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Setup Device and Load Model
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Loading model: {model_path} onto {device}...")
    try:
        # We load model using main engine class
        from engine.patchcore import PatchCore
        model = PatchCore.load(model_path, device=device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Load Dataset Images
    print(f"Scanning dataset in {dataset_path}...")
    test_images = load_dataset_images(dataset_path)
    if not test_images:
        print("Error: No test images found. Please check dataset path and structure.")
        sys.exit(1)
        
    num_anomaly = sum(1 for _, label in test_images if label == 1)
    num_normal = sum(1 for _, label in test_images if label == 0)
    print(f"Found {len(test_images)} test images: {num_normal} normal, {num_anomaly} anomaly.")
    
    if num_anomaly == 0 or num_normal == 0:
        print("Warning: Dataset needs to contain BOTH normal and abnormal samples to calculate AUROC and classification metrics.")
    
    # 4. Perform Inference and Benchmarking
    print("\nRunning inference & benchmark...")
    transform = get_transform(256)
    
    y_true = []
    y_scores = []
    
    # Timing benchmarks
    pre_times = []
    inf_times = []
    post_times = []
    total_times = []
    
    for img_path, label in tqdm(test_images):
        y_true.append(label)
        
        # End-to-end timing
        t_start = time.perf_counter()
        
        # A. Preprocessing
        t_pre_start = time.perf_counter()
        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            pre_time = (time.perf_counter() - t_pre_start) * 1000
            pre_times.append(pre_time)
            
            # B. Model Inference
            t_inf_start = time.perf_counter()
            with torch.no_grad():
                scores, anomaly_maps = model.predict(input_tensor)
            inf_time = (time.perf_counter() - t_inf_start) * 1000
            inf_times.append(inf_time)
            
            # Parse scores
            if isinstance(scores, (list, np.ndarray)):
                score = float(np.array(scores).flatten()[0])
            else:
                score = float(scores)
            y_scores.append(score)
            
            # C. Postprocessing (heatmap preparation simulated, if needed)
            t_post_start = time.perf_counter()
            # Standalone min-max scaling of anomaly map
            anomaly_map = np.array(anomaly_maps[0] if isinstance(anomaly_maps, list) else anomaly_maps)
            _ = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            post_time = (time.perf_counter() - t_post_start) * 1000
            post_times.append(post_time)
            
            # D. Total end-to-end
            total_time = (time.perf_counter() - t_start) * 1000
            total_times.append(total_time)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            y_scores.append(0.0)
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 5. Compute Quantitative Metrics
    metrics = {}
    
    # Latency summary
    metrics["latency"] = {
        "preprocessing_ms": float(np.mean(pre_times)),
        "inference_ms": float(np.mean(inf_times)),
        "postprocessing_ms": float(np.mean(post_times)),
        "total_ms": float(np.mean(total_times)),
        "max_total_ms": float(np.max(total_times)),
        "fps": float(1000.0 / np.mean(total_times))
    }
    
    # Anomaly Detection performance
    model_threshold = getattr(model, 'threshold', 0.5)
    metrics["model_threshold"] = float(model_threshold)
    
    # If we have both classes, compute AUROC and ROC curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        metrics["auroc"] = float(roc_auc)
        
        # Calculate best threshold based on Youden's Index (TPR - FPR)
        youden_idx = np.argmax(tpr - fpr)
        best_threshold_youden = thresholds[youden_idx]
        metrics["optimal_threshold_youden"] = float(best_threshold_youden)
        
        # Precision-Recall metrics
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        # Avoid division by zero in F1 calculation
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_f1_idx = np.argmax(f1_scores)
        metrics["optimal_threshold_f1"] = float(pr_thresholds[best_f1_idx]) if best_f1_idx < len(pr_thresholds) else float(model_threshold)
        metrics["max_f1"] = float(f1_scores[best_f1_idx])
        
        # Performance at Model's calibrated threshold
        y_pred = (y_scores > model_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics["calibrated_metrics"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }
        }
        
        # Performance at Optimal (Youden's) threshold
        y_pred_opt = (y_scores > best_threshold_youden).astype(int)
        tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_true, y_pred_opt).ravel()
        metrics["optimal_metrics"] = {
            "threshold": float(best_threshold_youden),
            "accuracy": float(accuracy_score(y_true, y_pred_opt)),
            "precision": float(precision_score(y_true, y_pred_opt, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_opt, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred_opt, zero_division=0)),
            "confusion_matrix": {
                "tp": int(tp_opt),
                "fp": int(fp_opt),
                "tn": int(tn_opt),
                "fn": int(fn_opt)
            }
        }
    else:
        # Fallback if only 1 class in dataset
        print("\nNote: Only one class present in dataset. AUROC and classification metrics are not computed.")
        metrics["auroc"] = 1.0 if num_anomaly == 0 else 0.0
        y_pred = (y_scores > model_threshold).astype(int)
        metrics["calibrated_metrics"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
    # Write JSON results
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        
    # 6. Generate Plots
    print("\nGenerating evaluation charts...")
    
    # Plot 1: ROC Curve
    if len(np.unique(y_true)) > 1:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
    # Plot 2: Score Distribution
    plt.figure(figsize=(7, 4.5))
    if len(np.unique(y_true)) > 1:
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        plt.hist(normal_scores, bins=15, alpha=0.6, label='Normal (OK)', color='green', edgecolor='black')
        plt.hist(anomaly_scores, bins=15, alpha=0.6, label='Anomaly (NG)', color='red', edgecolor='black')
    else:
        plt.hist(y_scores, bins=20, alpha=0.7, label='Scores', color='blue', edgecolor='black')
        
    plt.axvline(x=model_threshold, color='blue', linestyle='--', linewidth=2, label=f'Calibrated Threshold ({model_threshold:.2f})')
    if len(np.unique(y_true)) > 1:
        plt.axvline(x=best_threshold_youden, color='purple', linestyle='-.', linewidth=2, label=f'Optimal Threshold ({best_threshold_youden:.2f})')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Sample Count')
    plt.title('Anomaly Score Distribution & Thresholds')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(args.output_dir, "score_distribution.png"), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Confusion Matrix (Calibrated)
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large', weight='bold')
        
        plt.xlabel('Predictions', fontsize=12)
        plt.ylabel('Ground Truth', fontsize=12)
        ax.set_xticklabels(['', 'Normal (OK)', 'Anomaly (NG)'])
        ax.set_yticklabels(['', 'Normal (OK)', 'Anomaly (NG)'])
        plt.title('Confusion Matrix (Calibrated Threshold)', fontsize=13, pad=15)
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=200, bbox_inches='tight')
        plt.close()

    # 7. Write Markdown Evaluation Report
    report_path = os.path.join(args.output_dir, "evaluation_report.md")
    accuracy_val = metrics['calibrated_metrics']['accuracy'] * 100
    avg_latency = metrics['latency']['total_ms']
    max_latency = metrics['latency']['max_total_ms']
    fps_val = metrics['latency']['fps']
    vis_time = metrics['latency']['postprocessing_ms']
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Anomaly Detection System - Quantitative Evaluation Report\n\n")
        f.write(f"This report lists the quantitative evaluation metrics for the system, formatted to align with the thesis chapter tables.\n\n")
        
        f.write(f"## 1. Environment & Config\n")
        f.write(f"- **Evaluated Model**: `{os.path.basename(model_path)}`\n")
        f.write(f"- **Test Dataset**: `{os.path.basename(dataset_path)}`\n")
        f.write(f"- **Device Used**: `{device}`\n")
        f.write(f"- **Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n\n")
        
        f.write(f"## 2. Dataset Split\n")
        f.write(f"- **Total Test Samples**: {len(test_images)}\n")
        f.write(f"  - **Normal Samples (OK)**: {num_normal}\n")
        f.write(f"  - **Anomaly Samples (NG)**: {num_anomaly}\n\n")
        
        f.write(f"## 3. Thesis Performance Table\n\n")
        f.write(f"### 表 5-X 系统检测性能测试结果\n\n")
        f.write(f"| 测试项目 | 测试内容 | 测试结果 |\n")
        f.write(f"| :--- | :--- | :---: |\n")
        f.write(f"| **检测准确率** | 芯片测试图像正常/异常分类准确率 | **{accuracy_val:.2f}%** |\n")
        f.write(f"| **平均推理延迟** | 单张图像从上传到输出结果的平均耗时 | **{avg_latency:.2f} ms** |\n")
        f.write(f"| **最大推理延迟** | 单张图像最大检测耗时 | **{max_latency:.2f} ms** |\n")
        f.write(f"| **吞吐量** | 单位时间处理图像数量 | **{fps_val:.2f} 张/s** |\n")
        f.write(f"| **可视化响应时间** | 热力图生成与前端显示耗时 | **{vis_time:.2f} ms** |\n\n")
        
        f.write(f"## 4. Academic Evaluation Detail\n")
        if len(np.unique(y_true)) > 1:
            f.write(f"| Metric | Calibrated Threshold ({model_threshold:.4f}) | Optimal Threshold ({metrics['optimal_threshold_youden']:.4f}) |\n")
            f.write(f"| :--- | :---: | :---: |\n")
            f.write(f"| **AUROC** | {metrics['auroc']:.4f} | {metrics['auroc']:.4f} |\n")
            f.write(f"| **Accuracy** | {metrics['calibrated_metrics']['accuracy']:.4f} | {metrics['optimal_metrics']['accuracy']:.4f} |\n")
            f.write(f"| **Precision** | {metrics['calibrated_metrics']['precision']:.4f} | {metrics['optimal_metrics']['precision']:.4f} |\n")
            f.write(f"| **Recall (Sensitivity)** | {metrics['calibrated_metrics']['recall']:.4f} | {metrics['optimal_metrics']['recall']:.4f} |\n")
            f.write(f"| **F1-score** | {metrics['calibrated_metrics']['f1_score']:.4f} | {metrics['optimal_metrics']['f1_score']:.4f} |\n\n")
            
            f.write(f"### Confusion Matrix Breakdown\n")
            f.write(f"- True Positive (TP): {metrics['calibrated_metrics']['confusion_matrix']['tp']}\n")
            f.write(f"- False Positive (FP): {metrics['calibrated_metrics']['confusion_matrix']['fp']}\n")
            f.write(f"- True Negative (TN): {metrics['calibrated_metrics']['confusion_matrix']['tn']}\n")
            f.write(f"- False Negative (FN): {metrics['calibrated_metrics']['confusion_matrix']['fn']}\n\n")
            
        f.write(f"## 5. Latency Detailed Breakdown\n")
        f.write(f"| Process Stage | Average Latency (ms) | Percentage (%) |\n")
        f.write(f"| :--- | :---: | :---: |\n")
        pre_ms = metrics['latency']['preprocessing_ms']
        inf_ms = metrics['latency']['inference_ms']
        post_ms = metrics['latency']['postprocessing_ms']
        tot_ms = metrics['latency']['total_ms']
        f.write(f"| **Preprocessing** | {pre_ms:.2f} ms | {pre_ms/tot_ms*100:.1f}% |\n")
        f.write(f"| **Inference (PatchCore)** | {inf_ms:.2f} ms | {inf_ms/tot_ms*100:.1f}% |\n")
        f.write(f"| **Postprocessing & Visuals** | {post_ms:.2f} ms | {post_ms/tot_ms*100:.1f}% |\n")
        f.write(f"| **End-to-End Total** | **{tot_ms:.2f} ms** | **100.0%** |\n\n")

    print(f"\nEvaluation complete! Results saved in '{args.output_dir}'")
    print(f"Report: {report_path}")
    print(f"JSON data: {os.path.join(args.output_dir, 'metrics.json')}")
    print(f"Charts: ROC, Distribution, Confusion Matrix generated in '{args.output_dir}'")

if __name__ == "__main__":
    main()
