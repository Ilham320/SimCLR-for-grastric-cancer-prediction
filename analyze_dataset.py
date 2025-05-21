import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import ImageDataset
import yaml
import argparse
from collections import Counter
import sys

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def check_directory_exists(directory):
    """Check if directory exists and is accessible"""
    if not os.path.exists(directory):
        print(f"ERROR: Directory does not exist: {directory}")
        return False
    if not os.path.isdir(directory):
        print(f"ERROR: Path is not a directory: {directory}")
        return False
    return True

def analyze_dataset(data_dir, output_path=None):
    """Analyze dataset and create class distribution chart"""
    print(f"Analyzing dataset in: {data_dir}")
    
    # Check if directory exists
    if not check_directory_exists(data_dir):
        print("Exiting due to invalid dataset directory.")
        return None
    
    try:
        # Load dataset using ImageDataset class
        dataset = ImageDataset(data_dir, transform=None, return_labels=True)
        
        # Count samples per class
        labels = dataset.labels
        class_counts = Counter(labels)
        
        # Prepare data for plotting
        classes = dataset.classes
        counts = [class_counts[i] for i in range(len(classes))]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(classes, counts, color='royalblue')
        
        # Add count numbers on top of bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{count}', ha='center', va='bottom', fontsize=9)
        
        plt.title('Class Distribution in Histology Dataset', fontsize=16)
        plt.xlabel('Class', fontsize=14)
        plt.ylabel('Number of Samples', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        
        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution chart saved to: {output_path}")
        else:
            # Try to show the plot, but don't raise an exception if display is not available
            try:
                plt.show()
            except:
                print("Warning: Couldn't display the plot. Saving to default output file.")
                plt.savefig('dataset_class_distribution.png', dpi=300, bbox_inches='tight')
                print("Class distribution chart saved to: dataset_class_distribution.png")
        
        # Print additional statistics
        total_samples = len(dataset)
        print(f"Total samples: {total_samples}")
        print(f"Number of classes: {len(classes)}")
        print("Class distribution:")
        for i, cls in enumerate(classes):
            count = counts[i]
            percentage = (count / total_samples) * 100
            print(f"  {cls}: {count} samples ({percentage:.2f}%)")
        
        # Check for class imbalance
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
        
        return {
            'classes': classes,
            'counts': counts,
            'total_samples': total_samples,
            'imbalance_ratio': imbalance_ratio
        }
    
    except Exception as e:
        print(f"ERROR: Failed to analyze dataset: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset class distribution')
    parser.add_argument('--config', type=str, default='config/simclr_swin_resume_50epochs.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--output', type=str, default='dataset_class_distribution.png',
                        help='Output path for the class distribution chart')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset directory from config')
    parser.add_argument('--fallback_dir', type=str, default='data',
                        help='Fallback directory to use if config directory is not accessible')
    args = parser.parse_args()
    
    # Try to load configuration
    try:
        config = load_config(args.config)
        # Use provided dataset directory or get from config
        data_dir = args.dataset if args.dataset else config.get('dataset_dir')
    except Exception as e:
        print(f"WARNING: Failed to load config file: {str(e)}")
        data_dir = args.dataset
    
    # If data_dir is still None or doesn't exist, try the fallback
    if not data_dir or not check_directory_exists(data_dir):
        print(f"WARNING: Using fallback directory: {args.fallback_dir}")
        data_dir = args.fallback_dir
        
        # If fallback doesn't exist, attempt to find data directories in the workspace
        if not check_directory_exists(data_dir):
            possible_dirs = []
            for root, dirs, files in os.walk('.', topdown=True, followlinks=False):
                # Skip hidden directories and specific paths
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', '.venv']]
                
                for dir_name in dirs:
                    if 'data' in dir_name.lower() or 'dataset' in dir_name.lower():
                        full_path = os.path.join(root, dir_name)
                        possible_dirs.append(full_path)
            
            if possible_dirs:
                print("Found possible data directories:")
                for i, d in enumerate(possible_dirs):
                    print(f"{i+1}. {d}")
                
                print(f"Using the first found directory: {possible_dirs[0]}")
                data_dir = possible_dirs[0]
            else:
                print("ERROR: No valid data directory found. Please specify with --dataset")
                sys.exit(1)
    
    # Analyze dataset and create visualization
    analysis_result = analyze_dataset(data_dir, args.output)
    
    if analysis_result is None:
        print("Dataset analysis failed. Please check the provided paths and try again.")
        sys.exit(1)
    
    # Return success
    sys.exit(0)

if __name__ == "__main__":
    main()
