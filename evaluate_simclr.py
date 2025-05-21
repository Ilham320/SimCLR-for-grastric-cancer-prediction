import os
import torch
import yaml
import json
import numpy as np
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from simclr_model import SimCLR
from dataset import ImageDataset
from utils import linear_evaluate

def load_simclr_model(config_path, checkpoint_path):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Use the config from the checkpoint if available
    if 'config' in checkpoint:
        print("Using config from checkpoint")
        cfg = checkpoint['config']
    
    # Create model
    model = SimCLR(cfg)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}")
    else:
        print(f"Warning: model_state_dict not found in {checkpoint_path}")
    
    # Return only the encoder part for downstream tasks
    return model.encoder, cfg

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate SimCLR model')
    parser.add_argument('--best', action='store_true', help='Evaluate only the best model')
    parser.add_argument('--epoch', type=int, help='Evaluate a specific epoch')
    parser.add_argument('--checkpoint-path', type=str, help='Path to specific checkpoint')
    parser.add_argument('--config-path', type=str, help='Path to specific config file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load existing results if available
    results_file = 'simclr_evaluation_results.json'
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
            print(f"Loaded existing evaluation results from {results_file}")
    except FileNotFoundError:
        results = {}
        print(f"No existing results found. Creating new results file: {results_file}")
    
    # Use specific checkpoint and config if provided
    if args.checkpoint_path and args.config_path:
        if os.path.exists(args.checkpoint_path) and os.path.exists(args.config_path):
            print(f"Evaluating specific checkpoint: {args.checkpoint_path}")
            print(f"Using config from: {args.config_path}")
            simclr_checkpoints = [args.checkpoint_path]
            config_path = args.config_path
        else:
            print(f"Checkpoint or config file not found!")
            return
    else:
        # Get all checkpoint files
        checkpoints_dir = Path('checkpoints')
        config_path = 'config/simclr_config.yaml'
        
        # If evaluating a specific epoch
        if args.epoch is not None:
            specific_checkpoint = os.path.join('checkpoints', f'simclr_checkpoint_epoch_{args.epoch}.pt')
            if os.path.exists(specific_checkpoint):
                simclr_checkpoints = [specific_checkpoint]
                print(f"Evaluating only the checkpoint from epoch {args.epoch}")
            else:
                print(f"Checkpoint for epoch {args.epoch} not found!")
                return
        # If only evaluating best model
        elif args.best:
            best_model_path = os.path.join('checkpoints', 'simclr_best_model.pt')
            if os.path.exists(best_model_path):
                simclr_checkpoints = [best_model_path]
                print("Evaluating only the best model checkpoint")
            else:
                print("Best model checkpoint not found!")
                return
        # Otherwise get all checkpoints
        else:
            simclr_checkpoints = list(checkpoints_dir.glob('simclr_checkpoint_epoch_*.pt'))
            best_model_path = os.path.join('checkpoints', 'simclr_best_model.pt')
            if os.path.exists(best_model_path):
                simclr_checkpoints.append(Path(best_model_path))
    
    if not simclr_checkpoints:
        print("No SimCLR checkpoints found. Please train the model first using train_simclr.py")
        return
    
    # Sort checkpoints by epoch number
    def get_epoch_number(path):
        path_str = str(path)
        if 'best_model' in path_str:
            return float('inf')  # Best model appears last
        try:
            return int(path_str.split('_')[-1].split('.')[0])
        except:
            return float('inf')  # If we can't parse it, put it at the end
    
    simclr_checkpoints = sorted(simclr_checkpoints, key=get_epoch_number)
    
    # Setup for evaluation - use first checkpoint or the selected one for config
    checkpoint_to_use = str(simclr_checkpoints[0])
    print(f"Loading model from {checkpoint_to_use}")
    encoder, cfg = load_simclr_model(config_path, checkpoint_to_use)
    
    # Setup data transforms for evaluation
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup datasets
    train_path = cfg['train_eval_dir'] if 'train_eval_dir' in cfg else cfg['dataset_dir']
    test_path = cfg['test_eval_dir'] if 'test_eval_dir' in cfg else os.path.join(os.path.dirname(cfg['dataset_dir']), 'test')
    
    print(f"Training data: {train_path}")
    print(f"Test data: {test_path}")
    
    train_dataset = ImageDataset(
        train_path,
        transform=transform,
        return_labels=True,
        eval_mode=True
    )
    
    test_dataset = ImageDataset(
        test_path,
        transform=transform,
        return_labels=True,
        eval_mode=True
    )
    
    # Create dataloaders with reduced memory usage
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print("\nEvaluating SimCLR checkpoints...")
    
    for checkpoint_path in simclr_checkpoints:
        # Determine epoch number for output
        cp_str = str(checkpoint_path)
        if 'best_model' in cp_str:
            if 'swin' in cp_str:
                epoch_key = 'best_model_swin'
            else:
                epoch_key = 'best_model'
            print(f"\nEvaluating best model checkpoint: {cp_str}")
        else:
            try:
                epoch = get_epoch_number(checkpoint_path)
                epoch_key = str(epoch)
            except:
                epoch_key = os.path.basename(cp_str).replace('.pt', '')
            
            # Skip if already evaluated and not specifically requested
            if epoch_key in results and 'linear_accuracy' in results[epoch_key] and args.epoch is None and not args.checkpoint_path:
                print(f"Skipping epoch {epoch_key} (already evaluated)")
                continue
            
            print(f"\nEvaluating checkpoint: {epoch_key}")
        
        # Load model
        encoder, _ = load_simclr_model(config_path, cp_str)
        encoder = encoder.to(device)
        encoder.eval()
        
        # Linear evaluation
        print("Running linear evaluation...")
        linear_acc = linear_evaluate(encoder, train_loader, test_loader, device)
        
        # Save results
        epoch_results = results.get(epoch_key, {})
        epoch_results['linear_accuracy'] = float(linear_acc)
        results[epoch_key] = epoch_results
        
        print(f"Checkpoint {epoch_key}:")
        print(f"Linear Accuracy: {linear_acc:.2f}%")
        
        # Save results after each evaluation
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Clear GPU memory
        del encoder
        torch.cuda.empty_cache()
    
    # Print final results summary
    print("\nFinal Results:")
    print("Checkpoint\tLinear Accuracy")
    print("-" * 30)
    
    # Sort results by epoch number for display
    def sort_key(x):
        if x == 'best_model' or x == 'best_model_swin':
            return float('inf')
        try:
            return int(x)
        except:
            return x
            
    for epoch_key in sorted(results.keys(), key=sort_key):
        if 'linear_accuracy' in results[epoch_key]:
            print(f"{epoch_key}\t\t{results[epoch_key]['linear_accuracy']:.2f}%")

if __name__ == "__main__":
    main()