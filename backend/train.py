import os
import sys
import argparse
import torch

# Force absolute import path to find backend modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine.trainer import train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchCore Anomaly Detection Model via Command Line")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name inside backend/datasets or absolute path to dataset directory")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the output model file (e.g., my_model.pth). Defaults to <dataset_name>_model.pth")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", choices=["wide_resnet50_2", "resnet18"], help="Backbone feature extractor")
    parser.add_argument("--coreset_ratio", type=float, default=0.1, help="Coreset sampling ratio (0.01 to 1.0)")
    parser.add_argument("--n_neighbors", type=int, default=9, help="Number of nearest neighbors for KNN anomaly scoring")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu'). Auto-detected by default.")
    return parser.parse_args()

def progress_callback(stage, progress):
    """Prints training progress to the console"""
    print(f"[{progress}%] - Stage: {stage}")

def main():
    args = parse_args()
    
    # 1. Resolve Dataset Path
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        # Try looking inside backend/datasets
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", args.dataset)
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path '{args.dataset}' not found.")
            sys.exit(1)
            
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    
    # 2. Resolve Output Model Name
    model_name = args.model_name
    if not model_name:
        model_name = f"{dataset_name}_model.pth"
    if not model_name.endswith(".pth"):
        model_name += ".pth"
        
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 3. Setup Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print("="*60)
    print("                 PatchCore Model Training CLI")
    print("="*60)
    print(f"Dataset Path: {dataset_path}")
    print(f"Output Model: {os.path.join(models_dir, model_name)}")
    print(f"Backbone:     {args.backbone}")
    print(f"Coreset Ratio:{args.coreset_ratio}")
    print(f"K-Neighbors:  {args.n_neighbors}")
    print(f"Device:       {device}")
    print("="*60)
    
    # Force the engine model manager to use selected device
    from engine.model_manager import model_manager
    model_manager.device = device
    
    # 4. Start Training
    try:
        train_model(
            data_path=dataset_path,
            model_name=model_name,
            backbone_name=args.backbone,
            coreset_percentage=args.coreset_ratio,
            n_nearest_neighbour=args.n_neighbors,
            progress_callback=progress_callback
        )
        print("\n" + "="*60)
        print(f"Training successfully completed!")
        print(f"Model saved to models/{model_name}")
        print("="*60)
    except Exception as e:
        print(f"\nTraining Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
