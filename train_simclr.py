import os
import yaml
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import numpy as np
import copy
import argparse

from simclr_model import SimCLR, NT_Xent
from simclr_augmentations import HistologyTransform
from dataset import ImageDataset
import utils

def train_stage(config, stage_config=None, previous_model=None):
    """
    Train a single stage of SimCLR, either as standalone or as part of multi-stage training
    """
    # If stage_config is provided, override relevant parts of the main config
    if stage_config:
        # Create a copy of config and update with stage-specific settings
        stage_name = stage_config.get('name', 'unnamed_stage')
        print(f"\n{'='*80}\nStarting training stage: {stage_name}\n{'='*80}")
        
        # Override dataset paths if specified in stage config
        if 'dataset_dir' in stage_config:
            config['dataset_dir'] = stage_config['dataset_dir']
        if 'test_eval_dir' in stage_config:
            config['test_eval_dir'] = stage_config['test_eval_dir']
            
        # Override training parameters if specified
        if 'batch_size' in stage_config:
            config['batch_size'] = stage_config['batch_size']
        if 'epochs' in stage_config:
            config['epochs'] = stage_config['epochs']
        if 'temperature' in stage_config:
            config['temperature'] = stage_config['temperature']
            
        # Override optimizer settings if specified
        if 'optimizer' in stage_config:
            config['optimizer'].update(stage_config['optimizer'])
            
        # Custom checkpoint suffix for this stage
        checkpoint_suffix = stage_config.get('checkpoint_suffix', '')
    else:
        # Single-stage training
        stage_name = 'main'
        checkpoint_suffix = ''
    
    # Setup device and CUDA optimizations from config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Apply memory optimizations from config
    if config.get('memory_optimization', {}).get('cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
    
    # Enable gradient checkpointing if specified
    use_gradient_checkpointing = config.get('memory_optimization', {}).get('gradient_checkpointing', False)
    
    # Create histology-specific transforms with augmentation parameters from config
    transform = HistologyTransform(config)
    
    # Create dataset with train/validation split
    full_dataset = ImageDataset(config['dataset_dir'], transform=transform)
    
    # Create validation split (10% for validation)
    validation_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - validation_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    print(f"Dataset split: {train_size} training samples, {validation_size} validation samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config.get('memory_optimization', {}).get('pin_memory', True),
        persistent_workers=config['num_workers'] > 0,
        prefetch_factor=2 if config['num_workers'] > 0 else None,
        drop_last=True
    )
    
    # Create a dataloader for validation with standard transforms
    eval_transform = HistologyTransform(config, eval_mode=True)
    val_dataset.dataset.transform = eval_transform  # Replace transforms for validation
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    # Create datasets for linear evaluation
    from torchvision import transforms
    eval_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Setup datasets for linear evaluation
    train_path = config['train_eval_dir'] if 'train_eval_dir' in config else config['dataset_dir']
    test_path = config['test_eval_dir'] if 'test_eval_dir' in config else os.path.join(os.path.dirname(config['dataset_dir']), 'test')
    
    try:
        linear_train_dataset = ImageDataset(
            train_path,
            transform=eval_transform,
            return_labels=True,
            eval_mode=True
        )
        
        linear_test_dataset = ImageDataset(
            test_path,
            transform=eval_transform,
            return_labels=True,
            eval_mode=True
        )
        
        # Create dataloaders for linear evaluation
        linear_train_loader = DataLoader(
            linear_train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        linear_test_loader = DataLoader(
            linear_test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        linear_eval_available = True
    except Exception as e:
        print(f"Warning: Could not create linear evaluation datasets: {e}")
        print("Linear evaluation will be skipped")
        linear_eval_available = False
        linear_train_loader = None
        linear_test_loader = None
    
    # Setup checkpoint paths based on stage
    if checkpoint_suffix:
        best_checkpoint_path = os.path.join(config['checkpoint_dir'], f'simclr_{checkpoint_suffix}_best_model.pt')
    else:
        best_checkpoint_path = os.path.join(config['checkpoint_dir'], 'simclr_best_model.pt')
    
    old_projection_size = None
    start_epoch = 0
    best_loss = float('inf')
    best_accuracy = 0.0  # Track best validation accuracy
    
    # Check if checkpoint exists to get previous config
    if os.path.exists(best_checkpoint_path):
        print(f"Found checkpoint at {best_checkpoint_path}, checking configuration...")
        checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        
        # Check if we're using loss or accuracy for best model
        if 'accuracy' in checkpoint:
            best_accuracy = checkpoint['accuracy']
            print(f"Previous best validation accuracy: {best_accuracy:.2f}%")
        else:
            best_loss = checkpoint['loss']
            print(f"Previous best training loss: {best_loss:.4f}")
        
        if 'config' in checkpoint:
            old_projection_size = checkpoint['config'].get('projection_size', 256)
            print(f"Previous model used projection size: {old_projection_size}")
            print(f"Current config uses projection size: {config['projection_size']}")
    
    # Create SimCLR model with current config settings
    model = SimCLR(config).to(device)
    
    # For multi-stage training, transfer weights from the previous stage's model if provided
    if previous_model is not None:
        print(f"Transferring weights from previous stage model")
        # Only load encoder weights from previous model, not projection head
        encoder_state_dict = {k: v for k, v in previous_model.state_dict().items() 
                            if k.startswith('encoder.')}
        
        # Update current model's encoder weights
        model_dict = model.state_dict()
        model_dict.update(encoder_state_dict)
        model.load_state_dict(model_dict, strict=False)
        print("Successfully loaded encoder weights from previous stage")
    
    # Apply gradient checkpointing to backbone if enabled
    if use_gradient_checkpointing:
        print("Enabling gradient checkpointing for memory efficiency")
        if hasattr(model.encoder, 'layer4'):  # ResNet
            model.encoder.layer4.apply(lambda m: setattr(m, 'gradient_checkpointing', True))
        elif hasattr(model.encoder, 'blocks'):  # Vision Transformer / Swin Transformer
            for block in model.encoder.blocks:
                block.apply(lambda m: setattr(m, 'checkpoint', True) if hasattr(m, 'checkpoint') else None)
    
    # Initialize flag to know if we need to recreate optimizer later
    projection_size_changed = (old_projection_size is not None and 
                             old_projection_size != config['projection_size'])
    
    # Load weights when resuming training (if no previous model was provided)
    if previous_model is None and os.path.exists(best_checkpoint_path):
        print(f"Loading checkpoint from {best_checkpoint_path}...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        
        if projection_size_changed:
            print("Model architecture changed - loading only encoder weights")
            # Only load encoder weights, not projection head
            encoder_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                if not k.startswith('projector')}
            
            # Load partial state dict (encoder only)
            model_dict = model.state_dict()
            model_dict.update(encoder_state_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Successfully loaded encoder weights, randomly initialized new projection head")
        else:
            # Normal case - architecture unchanged, load all weights
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Resuming from epoch {start_epoch} with previous best accuracy: {best_accuracy:.2f}%")
    
    model.train()
    
    # Define function to evaluate model and save if it's the best
    def evaluate_and_save(epoch, train_loss):
        nonlocal best_accuracy
        
        if not linear_eval_available:
            print("Skipping linear evaluation (datasets not available)")
            return None
            
        print("\nEvaluating model with linear evaluation...")
        model.eval()
        
        # Make a copy of the encoder for evaluation
        encoder_copy = copy.deepcopy(model.encoder).to(device)
        
        # Run linear evaluation
        accuracy = utils.linear_evaluate(
            encoder_copy, 
            linear_train_loader, 
            linear_test_loader, 
            device,
            epochs=50,  # Reduced epochs for faster evaluation during training
            patience=5
        )
        
        # Free memory
        del encoder_copy
        torch.cuda.empty_cache()
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Linear Accuracy: {accuracy:.2f}%")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': train_loss,
            'accuracy': accuracy,
            'config': config
        }
        
        # Save regular epoch checkpoint
        if checkpoint_suffix:
            checkpoint_name = f'simclr_{checkpoint_suffix}_checkpoint_epoch_{epoch}.pt'
        else:
            checkpoint_name = f'simclr_checkpoint_epoch_{epoch}.pt'
            
        torch.save(checkpoint, os.path.join(config['checkpoint_dir'], checkpoint_name))
        
        # Save best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {accuracy:.2f}%, saving best model")
            torch.save(checkpoint, best_checkpoint_path)
        
        # Switch model back to training mode
        model.train()
        
        # Log results
        log_file = f'simclr_{stage_name}_training_log.txt' if stage_name != 'main' else 'simclr_training_log.txt'
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
        
        return accuracy
    
    # Create NT-Xent loss function with temperature from config
    criterion = NT_Xent(
        batch_size=config['batch_size'],
        temperature=config.get('temperature', 0.1)
    ).to(device)
    
    # Create optimizer with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
        eps=config['optimizer'].get('eps', 1e-8)
    )
    
    # Only load optimizer state if architecture hasn't changed and we're resuming the same stage
    if previous_model is None and os.path.exists(best_checkpoint_path) and 'optimizer_state_dict' in checkpoint and not projection_size_changed:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"Could not load optimizer state: {e}")
    else:
        print("Using fresh optimizer state")
    
    # Enhanced learning rate scheduler with warmup and cosine decay
    num_training_steps = len(train_loader) * config['epochs']
    num_warmup_steps = len(train_loader) * config['warmup_epochs']
    
    # Get scheduler parameters
    scheduler_config = config.get('scheduler', {})
    min_lr_factor = scheduler_config.get('final_lr_factor', 0.05)
    use_cosine = scheduler_config.get('cosine_decay', True)
    restart_epochs = scheduler_config.get('restart_epochs', [])
    restart_factor = scheduler_config.get('restart_factor', 0.5)
    
    # Convert restart epochs to steps
    restart_steps = [len(train_loader) * epoch for epoch in restart_epochs]
    
    # Improved cosine schedule with linear warmup and optional restarts
    def lr_lambda(current_step):
        # Check if we're at a restart point
        restart_lr_factor = 1.0
        for restart_step in restart_steps:
            if current_step >= restart_step:
                restart_lr_factor = restart_factor
                
        # Linear warmup phase
        if current_step < num_warmup_steps:
            return restart_lr_factor * float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay phase with min lr
        elif use_cosine:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return restart_lr_factor * max(min_lr_factor, cosine_decay)
        # Linear decay as fallback
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return restart_lr_factor * max(min_lr_factor, 1.0 - (1.0 - min_lr_factor) * progress)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Adjust scheduler to match start epoch if resuming
    if previous_model is None and start_epoch > 0:
        print(f"Adjusting scheduler to match start epoch {start_epoch}")
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()
    
    # Initialize gradient scaler for mixed precision
    use_fp16 = config.get('memory_optimization', {}).get('float16_precision', True)
    scaler = GradScaler(enabled=use_fp16)
    print(f"Mixed precision training: {'enabled' if use_fp16 else 'disabled'}")
    
    # Load scaler state if available and architecture hasn't changed
    if previous_model is None and os.path.exists(best_checkpoint_path) and 'scaler_state_dict' in checkpoint and use_fp16 and not projection_size_changed:
        try:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Loaded gradient scaler state from checkpoint")
        except Exception as e:
            print(f"Could not load gradient scaler state: {e}")
    
    # Configure gradient accumulation
    accum_steps = config.get('accumulate_grad_batches', 1)
    if accum_steps > 1:
        print(f"Using gradient accumulation with {accum_steps} steps")
    
    # Add memory management
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Early stopping setup
    patience = config.get('early_stopping_patience', 5)
    patience_counter = 0
    
    # Linear evaluation frequency
    linear_eval_freq = stage_config.get('linear_eval_freq', 5) if stage_config else 5
    
    print(f"Starting SimCLR training from epoch {start_epoch} for {config['epochs']} epochs...")
    
    # Function to evaluate model on validation set
    def validate_model(model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x in val_loader:
                # Check the format of the batch and handle appropriately
                if isinstance(x, list) or isinstance(x, tuple):
                    # It's already unpacked into a list/tuple of tensors
                    if len(x) == 2:  # Expected format with two views
                        x_i, x_j = x
                    else:
                        # If there are more than 2 elements, use first two
                        print(f"Warning: Expected 2 views but got {len(x)}. Using first two.")
                        x_i, x_j = x[0], x[1]
                else:
                    # It's a single tensor - in eval mode, just duplicate the images as both views
                    x_i = x.to(device, non_blocking=True)
                    x_j = x_i.clone()  # Use the same images for both views
                    
                    # For validation in eval mode, we can't do contrastive loss computation
                    # in the same way as training. Instead, we'll compute the representation quality
                    # by checking if the model produces consistent embeddings for the same image
                    with autocast(enabled=use_fp16):
                        # Pass each image through the encoder twice and compute consistency loss
                        _, _, z_i, z_j = model(x_i, x_j)
                        loss = criterion(z_i, z_j)
                    
                    val_loss += loss.item()
                    num_batches += 1
                    continue  # Skip the normal processing below as we've handled it here
                
                # Move data to device (for the list/tuple case)
                x_i = x_i.to(device, non_blocking=True)
                x_j = x_j.to(device, non_blocking=True)
                
                with autocast(enabled=use_fp16):
                    _, _, z_i, z_j = model(x_i, x_j)
                    loss = criterion(z_i, z_j)
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        model.train()
        return avg_val_loss
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        model.train()
        running_loss = 0.0
        
        # Clear memory before each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}") as pbar:
            optimizer.zero_grad(set_to_none=True)  # Initialize gradients
            
            for batch_idx, x in enumerate(pbar):
                # Check the format of the batch and handle appropriately
                if isinstance(x, list) or isinstance(x, tuple):
                    # It's already unpacked into a list/tuple of tensors
                    if len(x) == 2:  # Expected format with two views
                        x_i, x_j = x
                    else:
                        # If there are more than 2 elements, use first two
                        print(f"Warning: Expected 2 views but got {len(x)}. Using first two.")
                        x_i, x_j = x[0], x[1]
                else:
                    # It's a single tensor, might need to be split
                    if hasattr(x, 'shape') and len(x.shape) >= 4 and x.shape[0] % 2 == 0:
                        # Split batch in half to create two views
                        batch_size = x.shape[0] // 2
                        x_i = x[:batch_size]
                        x_j = x[batch_size:]
                    else:
                        raise ValueError(f"Could not extract two views from batch of type {type(x)} and shape {getattr(x, 'shape', 'unknown')}")
                
                # Move data to device
                x_i = x_i.to(device, non_blocking=True)
                x_j = x_j.to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast(enabled=use_fp16):
                    _, _, z_i, z_j = model(x_i, x_j)
                    loss = criterion(z_i, z_j) / accum_steps  # Scale loss for accumulation
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Update weights after accumulation steps or at the end of epoch
                if (batch_idx + 1) % accum_steps == 0 or batch_idx == len(train_loader) - 1:
                    # Unscale for gradient clipping
                    scaler.unscale_(optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Update LR scheduler
                    scheduler.step()
                
                # Calculate running loss (using the unscaled loss)
                running_loss += loss.item() * accum_steps  # Re-scale loss for logging
                avg_loss = running_loss / (batch_idx + 1)
                current_lr = scheduler.get_last_lr()[0]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
                # Clear cache periodically
                if (batch_idx + 1) % config.get('empty_cache_freq', 50) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Epoch completed, compute average loss
        epoch_loss = running_loss / len(train_loader)
        
        # Validate model
        print(f"Validating model after epoch {epoch+1}...")
        val_loss = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} completed: Train loss={epoch_loss:.4f}, Val loss={val_loss:.4f}")
        
        # Save model if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            print(f"New best validation loss: {best_loss:.4f}, saving model...")
            
            # Create checkpoint name based on stage
            if checkpoint_suffix:
                best_loss_checkpoint_path = os.path.join(config['checkpoint_dir'], f'simclr_{checkpoint_suffix}_best_loss_model.pt')
            else:
                best_loss_checkpoint_path = os.path.join(config['checkpoint_dir'], 'simclr_best_loss_model.pt')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
                'loss': best_loss,
                'config': config
            }, best_loss_checkpoint_path)
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
            # Regular checkpoint saving every few epochs
            if (epoch + 1) % 5 == 0:
                # Create checkpoint name based on stage
                if checkpoint_suffix:
                    checkpoint_path = os.path.join(config['checkpoint_dir'], f'simclr_{checkpoint_suffix}_checkpoint_epoch_{epoch+1}.pt')
                else:
                    checkpoint_path = os.path.join(config['checkpoint_dir'], f'simclr_checkpoint_epoch_{epoch+1}.pt')
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if use_fp16 else None,
                    'loss': epoch_loss,
                    'config': config
                }, checkpoint_path)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs without improvement")
            break
        
        # Evaluate and save model based on linear evaluation (periodically to save time)
        if (epoch + 1) % linear_eval_freq == 0 or epoch == config['epochs'] - 1:
            evaluate_and_save(epoch + 1, epoch_loss)
    
    print(f"Training stage {stage_name} completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    if linear_eval_available:
        print(f"Best validation accuracy: {best_accuracy:.2f}%")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train SimCLR model')
    parser.add_argument('--config', type=str, default='config/simclr_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    utils.set_seed(config['seed'])
    
    # Debug print
    print(f"SimCLR training started with config: {args.config}")
    print(f"Using backbone: {config.get('backbone', 'resnet50')}")
    
    # Check if we're doing multi-stage training
    if config.get('multi_stage_training', False):
        print("\nInitiating multi-stage training process")
        
        if 'stages' not in config or not config['stages']:
            print("Error: multi_stage_training is True but no stages defined in config")
            return
            
        # Train each stage sequentially
        previous_model = None
        for i, stage_config in enumerate(config['stages']):
            print(f"\nStarting stage {i+1}/{len(config['stages'])}: {stage_config.get('name', f'stage_{i+1}')}")
            previous_model = train_stage(config, stage_config, previous_model)
            
        print("\nAll stages completed successfully!")
        
    else:
        # Single stage training
        print("\nRunning single-stage training")
        train_stage(config)
    
    print("You can now evaluate the model using evaluate_simclr.py")


if __name__ == "__main__":
    main()