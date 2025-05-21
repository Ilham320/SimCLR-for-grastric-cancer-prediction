import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm

# Import your model and dataset
from simclr import SimCLR
from dataset import ImageDataset

def load_checkpoint(model_path, device):
    """Load model from checkpoint with proper handling of model architecture"""
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check what's in the checkpoint
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Extract config if available in checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config if not available
        config = {
            'model': 'resnet50',
            'projection_dim': 128,
            'num_classes': 8
        }
    
    print(f"Using config: {config}")
    
    # Create model instance based on your actual implementation
    model = SimCLR(
        base_model=config.get('backbone', 'resnet50'),
        out_dim=config.get('projection_size', 128),
        num_classes=8  # Hardcoded since it's specific to your dataset
    )
    
    # Try to load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the entire checkpoint is the state dict
        state_dict = checkpoint
    
    # Try to load with different prefixes/strategies
    try:
        # First attempt: direct loading
        model.load_state_dict(state_dict)
        print("Successfully loaded model state directly")
    except Exception as e1:
        print(f"Direct loading failed: {e1}")
        
        try:
            # Second attempt: handling encoder prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.'):
                    # SimCLR model has encoder as a sub-module
                    new_state_dict[k] = v
                else:
                    new_state_dict[f'encoder.{k}'] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            print("Loaded model with encoder prefix adjustment")
        except Exception as e2:
            print(f"Encoder prefix loading failed: {e2}")
            
            try:
                # Third attempt: load just the encoder part
                encoder_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('encoder.'):
                        encoder_state_dict[k[len('encoder.'):]] = v
                
                model.encoder.load_state_dict(encoder_state_dict, strict=False)
                print("Loaded just the encoder part of the model")
            except Exception as e3:
                print(f"Encoder-only loading failed: {e3}")
                print("Using model with random weights as fallback")
    
    model = model.to(device)
    model.eval()
    return model, config

def evaluate_model(model, dataloader, device, classes):
    """Evaluate model on the dataloader and return metrics and predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Add to lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class metrics
    class_metrics = []
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    for i, cls in enumerate(classes):
        class_metrics.append({
            'Class': cls,
            'Precision': precision_per_class[i],
            'Recall': recall_per_class[i],
            'F1 Score': f1_per_class[i],
            'Support': support_per_class[i]
        })
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_metrics': class_metrics,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

def save_results(results, classes, output_dir):
    """Save evaluation results as CSV files and plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall metrics to CSV
    metrics_df = pd.DataFrame({
        'Accuracy': [results['accuracy']],
        'Precision': [results['precision']],
        'Recall': [results['recall']],
        'F1 Score': [results['f1']]
    })
    metrics_csv_path = os.path.join(output_dir, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_csv_path)
    print(f"Saved performance metrics to {metrics_csv_path}")
    
    # Save per-class metrics to CSV
    class_df = pd.DataFrame(results['class_metrics'])
    class_csv_path = os.path.join(output_dir, 'class_metrics.csv')
    class_df.to_csv(class_csv_path, index=False)
    print(f"Saved class metrics to {class_csv_path}")
    
    # Create performance metrics plot
    plt.figure(figsize=(10, 6))
    metrics = {
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1 Score': results['f1']
    }
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1.0)
    for i, (k, v) in enumerate(metrics.items()):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
    plt.tight_layout()
    metrics_plot_path = os.path.join(output_dir, 'performance_metrics.png')
    plt.savefig(metrics_plot_path, dpi=300)
    plt.close()
    print(f"Saved performance metrics plot to {metrics_plot_path}")
    
    # Create confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {cm_plot_path}")

def main():
    # Model checkpoint path
    model_path = "C:/Users/mahmu/SimCLR/BYOL/checkpoints/simclr_best_model.pt"
    
    # Define output directory
    output_dir = "best_model_results"
    
    # Define class names
    classes = ['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    
    # Config path - if you have a config file
    config_path = "C:/Users/mahmu/SimCLR/BYOL/config/simclr_config.yaml"
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config if available
    config = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load model checkpoint
    model, checkpoint_config = load_checkpoint(model_path, device)
    
    # Get test dataset path
    test_dir = "C:/Users/mahmu/data/GCHTID/test"
    if config and 'test_eval_dir' in checkpoint_config:
        test_dir = checkpoint_config['test_eval_dir']
    
    print(f"Using test dataset: {test_dir}")
    
    # Define transforms for evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset and dataloader
    test_dataset = ImageDataset(
        data_dir=test_dir,  # Changed from root_dir to data_dir
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate model
    print("Evaluating model on test dataset...")
    results = evaluate_model(model, test_loader, device, classes)
    
    # Save results
    save_results(results, classes, output_dir)
    
    print(f"Evaluation complete. Results saved to {output_dir}")
    
    # Print summary of results
    print("\nPerformance Summary:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

if __name__ == "__main__":
    main()