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
    label_smoothing: float = 0.1,
    eval_every: int = 1,
    save_path: str = "checkpoints"):
    """Train the ConvE model."""
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_mrr = 0
    all_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_losses = []
        train_ranks = []
        train_hits = {1: 0, 3: 0, 10: 0}
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                # Move batch to device
                subject = batch['subject'].to(device)
                relation = batch['relation'].to(device)
                obj = batch['object'].to(device)
                
                # Forward pass
                scores = model(subject, relation)
                
                # Create target tensor with label smoothing
                n_entities = scores.size(1)
                targets = torch.zeros_like(scores).to(device)
                targets.scatter_(1, obj.unsqueeze(1), 1)
                targets = ((1.0 - label_smoothing) * targets) + (label_smoothing/n_entities)
                
                # Compute loss
                loss = F.binary_cross_entropy(scores, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar and track losses
                total_loss += loss.item()
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # After computing loss, calculate training metrics
                with torch.no_grad():
                    for i in range(scores.size(0)):
                        gold_score = scores[i, obj[i]]
                        rank = (scores[i] >= gold_score).sum().item()
                        train_ranks.append(rank)
                        
                        for k in train_hits.keys():
                            if rank <= k:
                                train_hits[k] += 1
                
        # Store average epoch loss
        avg_epoch_loss = np.mean(epoch_losses)
        all_losses.append(avg_epoch_loss)
        
        # Calculate training metrics
        train_mr = np.mean(train_ranks)
        train_mrr = np.mean([1.0 / r for r in train_ranks])
        total_examples = len(train_ranks)
        
        # Evaluate on validation set
        if (epoch + 1) % eval_every == 0:
            metrics = evaluate(model, valid_dataloader, device)
            
            # Update wandb logging to include training metrics
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': total_loss / len(train_dataloader),
                'train_loss_smooth': avg_epoch_loss,
                'train_loss_std': np.std(epoch_losses),
                'train_mr': train_mr,
                'train_mrr': train_mrr,
                'train_hits@1': train_hits[1]/total_examples,
                'train_hits@3': train_hits[3]/total_examples,
                'train_hits@10': train_hits[10]/total_examples,
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
            
            print(f"\nEpoch {epoch+1} Metrics:")
            print("Training:")
            print(f"Loss: {total_loss / len(train_dataloader):.4f}")
            print(f"MR: {train_mr:.1f}")
            print(f"MRR: {train_mrr:.4f}")
            print(f"Hits@1: {train_hits[1]/total_examples:.4f}")
            print(f"Hits@3: {train_hits[3]/total_examples:.4f}")
            print(f"Hits@10: {train_hits[10]/total_examples:.4f}")
            print("\nValidation:")
            print(f"Loss: {metrics['loss']:.4f}")
            print(f"MR: {metrics['mr']:.1f}")
            print(f"MRR: {metrics['mrr']:.4f}")
            print(f"Hits@1: {metrics['hits@1']:.4f}")
            print(f"Hits@3: {metrics['hits@3']:.4f}")
            print(f"Hits@10: {metrics['hits@10']:.4f}")
    
    return model
