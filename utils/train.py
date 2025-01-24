import torch
from torch.optim import Adam
import wandb
from tqdm import tqdm
from typing import Dict
import torch.nn.functional as F
import os
import numpy as np



@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    ranks = []
    hits = {1: 0, 3: 0, 10: 0}
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        subject = batch['subject'].to(device)
        relation = batch['relation'].to(device)
        object_true = batch['object'].to(device)
        filter_o = batch['filter_o'].to(device)
        
        scores = model(subject, relation)  # shape (B, num_entities)
        
        # Compute validation loss
        n_entities = scores.size(1)
        targets = torch.zeros_like(scores).to(device)
        targets.scatter_(1, object_true.unsqueeze(1), 1)
        targets = ((1.0 - 0.1) * targets) + (0.1/n_entities)  # Using default label smoothing of 0.1
        loss = F.binary_cross_entropy(scores, targets)
        total_loss += loss.item()
        
        # Compute ranks
        for i in range(scores.size(0)):
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            # filter out all known positives except the current gold
            mask[filter_o[i]] = False
            mask[object_true[i]] = True
            filtered_scores = scores[i].clone()
            filtered_scores[~mask] = float('-inf')
            
            gold_score = filtered_scores[object_true[i]]
            rank = (filtered_scores >= gold_score).sum().item()
            ranks.append(rank)
            
            for k in hits.keys():
                if rank <= k:
                    hits[k] += 1

    mr = np.mean(ranks)
    mrr = np.mean([1.0 / r for r in ranks])
    total = len(ranks)
        
    metrics = {
        'mr': mr,
        'mrr': mrr,
        'hits@1': hits[1]/total,
        'hits@3': hits[3]/total,
        'hits@10': hits[10]/total,
        'loss': total_loss / len(dataloader)  # Add average validation loss
    }
    return metrics

def train_conve(
    model,
    train_dataloader,
    valid_dataloader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda",
    eval_every: int = 1,
    save_path: str = "checkpoints"):
    """Train the ConvE model."""
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Device being used: {device}")
    print(f"Training samples per epoch: {len(train_dataloader.dataset):,}")
    print(f"Validation samples: {len(valid_dataloader.dataset):,}")
    
    # Training loop
    best_mrr = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_losses = []
        batch_scores = []  # Store prediction scores
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                subject = batch['subject'].to(device)
                relation = batch['relation'].to(device)
                object_true = batch['object'].to(device)
                
                # Forward pass to get all scores
                scores = model(subject, relation)  # shape: [batch_size, num_entities]
                
                # Debug prints for first batch of first epoch
                if epoch == 0 and batch_idx == 0:
                    print("\nFirst batch debug info:")
                    print(f"Scores shape: {scores.shape}")
                    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
                    print(f"Mean score: {scores.mean():.4f}")
                    batch_scores.append(scores.detach().cpu().numpy())
                
                # Binary Cross Entropy loss with label smoothing
                n_entities = scores.size(1)
                targets = torch.zeros_like(scores).to(device)
                targets.scatter_(1, object_true.unsqueeze(1), 1)
                targets = ((1.0 - 0.1) * targets) + (0.1/n_entities)  # Label smoothing of 0.1
                
                if epoch == 0 and batch_idx == 0:
                    print(f"Targets shape: {targets.shape}")
                    print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
                    #print(f"Number of positive targets per row: {(targets == 0.9).sum(1).mean():.2f}")
                
                loss = F.binary_cross_entropy(scores, targets)
                
                if epoch == 0 and batch_idx == 0:
                    print(f"Initial loss: {loss.item():.4f}")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                epoch_losses.append(loss.item())
                
                # Detailed metrics every 100 batches
                if batch_idx % 100 == 0:
                    current_mean_loss = np.mean(epoch_losses[-100:]) if len(epoch_losses) >= 100 else np.mean(epoch_losses)
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss_100': f"{current_mean_loss:.4f}"
                    })
        
        # Calculate average epoch loss
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1} Training Statistics:")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        print(f"Loss Range: [{min(epoch_losses):.4f}, {max(epoch_losses):.4f}]")
        
        # Evaluate on validation set
        if (epoch + 1) % eval_every == 0:
            metrics = evaluate(model, valid_dataloader, device)
            
            print("\nValidation Statistics:")
            print(f"Scores from first batch (mean/std): {np.mean(batch_scores):.4f}/{np.std(batch_scores):.4f}")
            print(f"Loss: {metrics['loss']:.4f}")
            print(f"MR: {metrics['mr']:.1f}")
            print(f"MRR: {metrics['mrr']:.4f}")
            print(f"Hits@1: {metrics['hits@1']:.4f}")
            print(f"Hits@3: {metrics['hits@3']:.4f}")
            print(f"Hits@10: {metrics['hits@10']:.4f}")
            
            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'valid_loss': metrics['loss'],
                'valid_mr': metrics['mr'],
                'valid_mrr': metrics['mrr'],
                'valid_hits@1': metrics['hits@1'],
                'valid_hits@3': metrics['hits@3'],
                'valid_hits@10': metrics['hits@10'],
            })
            
            # Save best model
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mrr': best_mrr,
                }, f"{save_path}/best_model.pt")
                print(f"\nNew best MRR: {best_mrr:.4f} - Saved model checkpoint")
    
    return model