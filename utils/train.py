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
        
        # Get raw scores (logits)
        scores = model(subject, relation)  # shape (B, num_entities)
        
        # Compute validation loss with BCE logits
        n_entities = scores.size(1)
        targets = torch.zeros_like(scores).to(device)
        targets.scatter_(1, object_true.unsqueeze(1), 1)
        targets = ((1.0 - 0.1) * targets) + (0.1/n_entities)
        
        # Use BCE with logits like in training
        loss = F.binary_cross_entropy_with_logits(scores, targets)
        total_loss += loss.item()
        
        # Convert to probabilities for ranking
        scores = torch.sigmoid(scores)
        
        # Compute ranks
        for i in range(scores.size(0)):
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            mask[filter_o[i]] = False
            mask[object_true[i]] = True
            
            filtered_scores = scores[i].clone()
            filtered_scores[~mask] = float('-inf')
            
            # Get score of true object
            gold_score = filtered_scores[object_true[i]]
            
            # Calculate rank (number of scores >= gold_score)
            rank = (filtered_scores >= gold_score).sum().item()
            ranks.append(rank)
            
            # Update Hits@k
            for k in hits.keys():
                if rank <= k:
                    hits[k] += 1
    
    # Calculate metrics
    mr = np.mean(ranks)
    mrr = np.mean([1.0 / r for r in ranks])
    
    metrics = {
        'mr': mr,
        'mrr': mrr,
        'hits@1': hits[1]/len(ranks),
        'hits@3': hits[3]/len(ranks),
        'hits@10': hits[10]/len(ranks),
        'loss': total_loss / len(dataloader)
    }
    
    return metrics


@torch.no_grad()
def evaluate_2pass(model, dataloader, device):
    model.eval()

    # We'll collect ranks from both head and tail passes together
    ranks_all = []
    hits_all = {1: 0, 3: 0, 10: 0}

    # Utility to update ranks, hits@k
    def update_ranks(filtered_scores, gold_idx, ranks_list, hits_dict):
        gold_score = filtered_scores[gold_idx]
        # rank = number of scores >= gold_score
        rank = (filtered_scores >= gold_score).sum().item()
        ranks_list.append(rank)
        for k in hits_dict.keys():
            if rank <= k:
                hits_dict[k] += 1

    total_loss = 0
    for batch in tqdm(dataloader, desc="Evaluating Head+Tail"):
        subject = batch['subject'].to(device)
        relation = batch['relation'].to(device)
        object_true = batch['object'].to(device)
        filter_o = batch['filter_o'].to(device)
        filter_h = batch['filter_h'].to(device)

        # Compute tail prediction loss
        scores_tail = model.forward(subject, relation)
        n_entities = scores_tail.size(1)
        targets_tail = torch.zeros_like(scores_tail).to(device)
        targets_tail.scatter_(1, object_true.unsqueeze(1), 1)
        targets_tail = ((1.0 - 0.1) * targets_tail) + (0.1/n_entities)
        loss_tail = F.binary_cross_entropy_with_logits(scores_tail, targets_tail)

        # Compute head prediction loss
        scores_head = model.forward_head(object_true, relation)
        targets_head = torch.zeros_like(scores_head).to(device)
        targets_head.scatter_(1, subject.unsqueeze(1), 1)
        targets_head = ((1.0 - 0.1) * targets_head) + (0.1/n_entities)
        loss_head = F.binary_cross_entropy_with_logits(scores_head, targets_head)

        # Average the losses
        total_loss += (loss_tail.item() + loss_head.item()) / 2

        # ===== Tail mode (like your existing code) =====
        # shape = (B, num_entities)
        scores_tail = scores_tail.clone()
        batch_size, num_ents = scores_tail.size()

        for i in range(batch_size):
            # Zero out or -inf all known tails in 'filter_o[i]'
            # except the true object at position object_true[i]
            # We'll do the standard "filtered" approach:
            mask = torch.ones(num_ents, dtype=torch.bool, device=device)
            # filter out known positives
            mask[filter_o[i]] = False
            # re-include the gold tail
            mask[object_true[i]] = True
            # anything not in mask -> -inf
            scores_tail[i][~mask] = float('-inf')

        # Now compute ranks for each item i in the batch
        for i in range(batch_size):
            update_ranks(
                scores_tail[i],
                object_true[i].item(),
                ranks_all,
                hits_all
            )

        # ===== Head mode (the new step) =====
        # shape = (B, num_entities)
        scores_head = scores_head.clone()
        for i in range(batch_size):
            mask = torch.ones(num_ents, dtype=torch.bool, device=device)
            mask[filter_h[i]] = False
            # re-include the gold head
            mask[subject[i]] = True
            scores_head[i][~mask] = float('-inf')

        for i in range(batch_size):
            update_ranks(
                scores_head[i],
                subject[i].item(),
                ranks_all,
                hits_all
            )

    # Now compute "final" metrics over head+tail
    mr = np.mean(ranks_all)
    mrr = np.mean([1.0 / r for r in ranks_all])
    hits_at_1 = hits_all[1] / len(ranks_all)
    hits_at_3 = hits_all[3] / len(ranks_all)
    hits_at_10 = hits_all[10] / len(ranks_all)

    metrics = {
        'mr': mr,
        'mrr': mrr,
        'hits@1': hits_at_1,
        'hits@3': hits_at_3,
        'hits@10': hits_at_10,
        'loss': total_loss / len(dataloader)
    }

    return metrics

def train_conve(
    model,
    train_dataloader,
    valid_dataloader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    eval_every: int = 1,
    save_path: str = "checkpoints"):
    """Train the ConvE model with GPU optimizations if available, fallback to CPU if not."""
    
    # Move model to device
    model = model.to(device)
    
    # CUDA-specific optimizations
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        train_dataloader.pin_memory = True
        valid_dataloader.pin_memory = True
        optimizer = Adam(model.parameters(), lr=learning_rate, fused=True)
        scaler = torch.amp.GradScaler()
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate)
        scaler = None
    
    # Print setup info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Device being used: {device}")
    print(f"Training samples per epoch: {len(train_dataloader.dataset):,}")
    print(f"Validation samples: {len(valid_dataloader.dataset):,}")
    
    best_mrr = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_losses = []
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                subject = batch['subject'].to(device, non_blocking=True if device=="cuda" else False)
                relation = batch['relation'].to(device, non_blocking=True if device=="cuda" else False)
                object_true = batch['object'].to(device, non_blocking=True if device=="cuda" else False)
                filter_o = batch['filter_o'].to(device, non_blocking=True if device=="cuda" else False)  # Get filters
                
                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        scores = model(subject, relation)
                        n_entities = scores.size(1)
                        
                        # Create target tensor with filtered negatives
                        targets = torch.zeros_like(scores, device=device)
                        # Set all valid answers (including filtered ones) to 1
                        for i in range(scores.size(0)):
                            targets[i][filter_o[i]] = 1
                        
                        # Apply label smoothing only to real negatives
                        neg_mask = targets == 0
                        pos_mask = targets == 1
                        targets = targets.clone()
                        targets[neg_mask] = 0.1/n_entities
                        targets[pos_mask] = 1.0 - 0.1
                        
                        loss = F.binary_cross_entropy_with_logits(scores, targets)
                else:
                    # CPU version (same logic)
                    scores = model(subject, relation)
                    n_entities = scores.size(1)
                    targets = torch.zeros_like(scores, device=device)
                    for i in range(scores.size(0)):
                        targets[i][filter_o[i]] = 1
                    
                    neg_mask = targets == 0
                    pos_mask = targets == 1
                    targets = targets.clone()
                    targets[neg_mask] = 0.1/n_entities
                    targets[pos_mask] = 1.0 - 0.1
                    
                    loss = F.binary_cross_entropy_with_logits(scores, targets)
                
                # Backward pass with or without mixed precision
                optimizer.zero_grad(set_to_none=True)
                if device == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                current_loss = total_loss / (batch_idx + 1)
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})

        
        # Memory cleanup for CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Evaluation phase
        if (epoch + 1) % 1 == 0:  # Evaluate every epoch
            model.eval()
            with torch.cuda.amp.autocast():  # Use mixed precision for evaluation too
                #metrics = evaluate(model, valid_dataloader, device)
                metrics = evaluate_2pass(model, valid_dataloader, device)
            
            print("\nValidation Statistics:")
            print(f"Loss: {metrics['loss']:.4f}")
            print(f"MR: {metrics['mr']:.1f}")
            print(f"MRR: {metrics['mrr']:.4f}")
            print(f"Hits@1: {metrics['hits@1']:.4f}")
            print(f"Hits@3: {metrics['hits@3']:.4f}")
            print(f"Hits@10: {metrics['hits@10']:.4f}")
            
            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': np.mean(epoch_losses),
                'valid_loss': metrics['loss'],
                'valid_mr': metrics['mr'],
                'valid_mrr': metrics['mrr'],
                'valid_hits@1': metrics['hits@1'],
                'valid_hits@3': metrics['hits@3'],
                'valid_hits@10': metrics['hits@10'],
            })
            
            # # Save best model (asynchronously)
            # if metrics['mrr'] > best_mrr:
            #     best_mrr = metrics['mrr']
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'best_mrr': best_mrr,
            #     }, f"checkpoints/best_model.pt", _use_new_zipfile_serialization=True)
    
    return model