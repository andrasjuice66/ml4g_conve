import torch
from torch.optim import Adam
import wandb
from tqdm import tqdm
from typing import Dict
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, dataloader, device) -> Dict[str, float]:
    """Evaluate the model on the given dataloader."""
    model.eval()
    
    ranks = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_10 = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move batch to device
        subject = batch['subject'].to(device)
        relation = batch['relation'].to(device)
        obj = batch['object'].to(device)
        filter_out = batch['filter_out'].to(device)
        
        # Get scores for all possible objects
        scores = model(subject, relation)
        
        # Get target scores
        target_scores = scores.gather(1, obj.unsqueeze(1))
        
        # Filter out known positive triples
        for i in range(scores.shape[0]):
            scores[i, filter_out[i, :]] = float('-inf')
        
        # Compute rank
        ranks_batch = (scores >= target_scores).sum(dim=1)
        
        ranks.extend(ranks_batch.cpu().tolist())
        hits_at_1.extend((ranks_batch <= 1).float().cpu().tolist())
        hits_at_3.extend((ranks_batch <= 3).float().cpu().tolist())
        hits_at_10.extend((ranks_batch <= 10).float().cpu().tolist())
    
    return {
        'mrr': sum(1/r for r in ranks) / len(ranks),
        'hits@1': sum(hits_at_1) / len(hits_at_1),
        'hits@3': sum(hits_at_3) / len(hits_at_3),
        'hits@10': sum(hits_at_10) / len(hits_at_10),
    }

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
            print(f"MRR: {metrics['mrr']:.4f}")
            print(f"Hits@1: {metrics['hits@1']:.4f}")
            print(f"Hits@3: {metrics['hits@3']:.4f}")
            print(f"Hits@10: {metrics['hits@10']:.4f}")
    
    return model