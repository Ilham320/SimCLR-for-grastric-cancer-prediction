import torch
import yaml
from torchvision import transforms
from dataset import ImageDataset
from utils import linear_evaluate
import timm
import torch.nn as nn
from pathlib import Path
import json
import os

def load_model(config_path, checkpoint_path):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load checkpoint first to get any config updates saved with the model
    checkpoint = torch.load(checkpoint_path)
    
    # Use the config from the checkpoint if available
    if 'config' in checkpoint:
        print("Using config from checkpoint")
        cfg = checkpoint['config']
    
    # Import our custom BYOL implementation
    from models import BYOL
    
    # Create model using our implementation
    model = BYOL(cfg)
    
    # Load checkpoint
    if 'model_state_dict' in checkpoint:
        print("Loading model_state_dict from checkpoint")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: model_state_dict not found in checkpoint")
    
    # Return the online encoder part of the model
    return model.online_encoder, cfg

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load existing results if available
    try:
        with open('evaluation_results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    
    # Get all checkpoint files sorted by epoch number
    checkpoints_dir = Path('checkpoints')
    checkpoint_files = sorted(
        [f for f in checkpoints_dir.glob('checkpoint_epoch_*.pt')],
        key=lambda x: int(x.stem.split('_')[-1])
    )
    
    # If no epoch checkpoints found, check for best_model.pt
    if not checkpoint_files and os.path.exists('checkpoints/best_model.pt'):
        print("No epoch checkpoints found, using best_model.pt")
        checkpoint_files = ['checkpoints/best_model.pt']
        
    if not checkpoint_files:
        print("No checkpoint files found. Please make sure training has completed at least one epoch.")
        return
    
    # Load model to get config and setup data
    model_path = checkpoint_files[0]
    print(f"Loading model from {model_path}")
    first_model, cfg = load_model('config/byol_config.yaml', model_path)
    
    # Setup datasets with correct paths - don't append 'train' or 'test' if already in the path
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_path = cfg['train_eval_dir'] if 'train_eval_dir' in cfg else cfg['dataset_dir']
    test_path = cfg['test_eval_dir'] if 'test_eval_dir' in cfg else os.path.join(os.path.dirname(cfg['dataset_dir']), 'test')
    
    print(f"Using train data from: {train_path}")
    print(f"Using test data from: {test_path}")
    
    train_dataset = ImageDataset(
        train_path, 
        transform=transform,
        return_labels=True
    )
    
    test_dataset = ImageDataset(
        test_path,
        transform=transform,
        return_labels=True
    )
    
    # Modified DataLoader settings for Windows compatibility
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=128, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("\nEvaluating checkpoints...")
    for checkpoint_path in checkpoint_files:
        epoch = int(checkpoint_path.stem.split('_')[-1]) if 'checkpoint_epoch_' in checkpoint_path else None
        
        # Skip epochs before 11
        if epoch is not None and epoch < 11:
            continue
            
        if epoch is not None and epoch > 26:  # Only evaluate up to epoch 26
            break
            
        # Skip if already evaluated
        if epoch is not None and str(epoch) in results and 'linear_accuracy' in results[str(epoch)]:
            print(f"Skipping epoch {epoch} (already evaluated)")
            continue
            
        print(f"\nEvaluating checkpoint from epoch {epoch if epoch is not None else 'best_model.pt'}")
        model, _ = load_model('config/byol_config.yaml', checkpoint_path)
        model = model.to(device)
        model.eval()
        
        # Evaluate using Linear classifier
        print("Running Linear evaluation...")
        linear_acc = linear_evaluate(model, train_loader, test_loader, device)
        
        # Preserve existing KNN accuracy if available
        epoch_results = results.get(str(epoch) if epoch is not None else 'best_model', {})
        epoch_results['linear_accuracy'] = float(linear_acc)
        results[str(epoch) if epoch is not None else 'best_model'] = epoch_results
        
        print(f"Epoch {epoch if epoch is not None else 'best_model.pt'}:")
        print(f"Linear Accuracy: {linear_acc:.2f}%")
        
        # Save results after each epoch
        with open('evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    print("\nFinal Results:")
    print("Epoch\tLinear Acc")
    print("-" * 20)
    for epoch in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
        if epoch.isdigit() and int(epoch) >= 11:  # Only show results from epoch 11 onwards
            print(f"{epoch}\t{results[epoch]['linear_accuracy']:.2f}%")
        elif epoch == 'best_model':
            print(f"{epoch}\t{results[epoch]['linear_accuracy']:.2f}%")

if __name__ == '__main__':
    main()