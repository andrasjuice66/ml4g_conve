import torch
from torch.optim import Adam
import wandb
from tqdm import tqdm
from typing import Dict
import torch.nn.functional as F
import os

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    ranks_left = []
    ranks_right = []
    hits = {1: [], 3: [], 10: []}
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        subject = batch['subject'].to(device)
        relation = batch['relation'].to(device)
        object_true = batch['object'].to(device)
        filter_o = batch['filter_o'].to(device)
        filter_s = batch['filter_s'].to(device)
        
        # Forward direction (object corruption)
        scores1 = model(subject, relation)
        # Reverse direction (subject corruption) - use same relation
        scores2 = model(object_true, relation)  # Using same relation, model handles direction
        
        for idx, (s, r, o) in enumerate(zip(subject, relation, object_true)):
            # Object corruption
            mask1 = torch.ones_like(scores1[idx], dtype=torch.bool)
            mask1[filter_o[idx]] = False  # Apply forward filters
            mask1[o] = True
            filtered_scores1 = scores1[idx].clone()
            filtered_scores1[~mask1] = float('-inf')
            rank1 = (filtered_scores1 >= filtered_scores1[o]).sum().item()
            
            # Subject corruption
            mask2 = torch.ones_like(scores2[idx], dtype=torch.bool)
            mask2[filter_s[idx]] = False  # Apply reverse filters
            mask2[s] = True
            filtered_scores2 = scores2[idx].clone()
            filtered_scores2[~mask2] = float('-inf')
            rank2 = (filtered_scores2 >= filtered_scores2[s]).sum().item()
            
            ranks_left.append(rank1)
            ranks_right.append(rank2)
            
            for k in hits.keys():
                hits[k].append(1 if rank1 <= k else 0)
                hits[k].append(1 if rank2 <= k else 0)

    metrics = {
        'mr': (sum(ranks_left) + sum(ranks_right)) / (len(ranks_left) + len(ranks_right)),
        'mrr': sum(1/r for r in ranks_left + ranks_right) / (len(ranks_left) + len(ranks_right)),
        'hits@1': sum(hits[1]) / len(hits[1]),
        'hits@3': sum(hits[3]) / len(hits[3]),
        'hits@10': sum(hits[10]) / len(hits[10])
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
    save_path: str = "checkpoints"
):
    """Train the ConvE model."""
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_mrr = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
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
                
                # Update progress bar
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Evaluate on validation set
        if (epoch + 1) % eval_every == 0:
            metrics = evaluate(model, valid_dataloader, device)
            
            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': total_loss / len(train_dataloader),
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
            
            print(f"\nEpoch {epoch+1} Validation Metrics:")
            print(f"MR: {metrics['mr']:.1f}")
            print(f"MRR: {metrics['mrr']:.4f}")
            print(f"Hits@1: {metrics['hits@1']:.4f}")
            print(f"Hits@3: {metrics['hits@3']:.4f}")
            print(f"Hits@10: {metrics['hits@10']:.4f}")
    
    return model