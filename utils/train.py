# utils/train.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import wandb

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Single-pass evaluator: tail prediction only.
    For each (h, r), we filter out known positives except the real tail.
    Then compute rank and hits@k.
    """
    model.eval()
    ranks = []
    hits = {1: 0, 3: 0, 10: 0}
    total_loss = 0

    for batch in tqdm(dataloader, desc="Evaluating (Tail Only)"):
        subject = batch['subject'].to(device)
        relation = batch['relation'].to(device)
        object_true = batch['object'].to(device)
        filter_o = batch['filter_o'].to(device)

        # Get raw scores
        scores = model(subject, relation)  # (B, num_entities)
        n_entities = scores.size(1)

        # Build multi-hot target for BCE
        targets = torch.zeros_like(scores, device=device)
        for i in range(scores.size(0)):
            # Mark all known positives for (h, r)
            targets[i][filter_o[i]] = 1

        # Label smoothing
        neg_mask = targets == 0
        pos_mask = targets == 1
        targets = targets.clone()
        targets[neg_mask] = 0.1/n_entities
        targets[pos_mask] = 1.0 - 0.1

        loss = F.binary_cross_entropy_with_logits(scores, targets)
        total_loss += loss.item()

        # Filtered ranking evaluation
        scores = torch.sigmoid(scores)
        for i in range(scores.size(0)):
            # Filter out known tails (set them -inf) except the gold object
            mask = torch.ones(n_entities, dtype=torch.bool, device=device)
            mask[filter_o[i]] = False
            # Re-include the gold object
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
    """
    Two-pass evaluator: tail and head prediction.
    """
    model.eval()
    ranks_all = []
    hits_all = {1: 0, 3: 0, 10: 0}
    total_loss = 0

    def update_ranks(scores, true_idx, filter_idx, ranks_list, hits_dict):
        # Create binary mask for filtering
        mask = torch.ones(scores.size(0), dtype=torch.bool, device=device)
        mask[filter_idx] = False  # Remove all known positives
        mask[true_idx] = True    # Add back the target triple
        
        # Apply filter
        filtered_scores = scores.clone()
        filtered_scores[~mask] = float('-inf')
        
        # Get rank
        true_score = filtered_scores[true_idx]
        rank = (filtered_scores >= true_score).sum().item()
        
        ranks_list.append(rank)
        for k in hits_dict.keys():
            if rank <= k:
                hits_dict[k] += 1

    for batch in tqdm(dataloader, desc="Evaluating (Head+Tail)"):
        subject = batch['subject'].to(device)
        relation = batch['relation'].to(device)
        object_true = batch['object'].to(device)
        filter_o = batch['filter_o'].to(device)
        filter_h = batch['filter_h'].to(device)

        # Tail prediction
        scores_tail = model(subject, relation)
        scores_tail = torch.sigmoid(scores_tail)
        for i in range(scores_tail.size(0)):
            update_ranks(
                scores_tail[i], 
                object_true[i].item(),
                filter_o[i],
                ranks_all,
                hits_all
            )

        # Head prediction 
        scores_head = model.forward_head(object_true, relation)
        scores_head = torch.sigmoid(scores_head)
        for i in range(scores_head.size(0)):
            update_ranks(
                scores_head[i],
                subject[i].item(),
                filter_h[i], 
                ranks_all,
                hits_all
            )

    # Calculate metrics
    mr = np.mean(ranks_all)
    mrr = np.mean([1.0 / r for r in ranks_all])
    
    metrics = {
        'mr': mr,
        'mrr': mrr,
        'hits@1': hits_all[1]/len(ranks_all),
        'hits@3': hits_all[3]/len(ranks_all),
        'hits@10': hits_all[10]/len(ranks_all),
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
    save_path: str = "checkpoints"
):
    """
    Main training loop for the ConvE model.
    Uses multi-label target (via filter_o) and label smoothing on negatives.
    """
    model = model.to(device)

    # GPU-specific optimizations if available
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        train_dataloader.pin_memory = True
        valid_dataloader.pin_memory = True
        optimizer = Adam(model.parameters(), lr=learning_rate, fused=True)
        scaler = torch.cuda.amp.GradScaler()
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate)
        scaler = None

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataloader.dataset):,}")
    print(f"Validation samples: {len(valid_dataloader.dataset):,}")

    best_mrr = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_losses = []

        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                subject = batch['subject'].to(device)
                relation = batch['relation'].to(device)
                filter_o = batch['filter_o'].to(device)

                # Forward pass
                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        scores = model(subject, relation)
                        n_entities = scores.size(1)

                        # Build multi-hot target
                        targets = torch.zeros_like(scores, device=device)
                        for i in range(scores.size(0)):
                            targets[i][filter_o[i]] = 1

                        # Label smoothing
                        neg_mask = targets == 0
                        pos_mask = targets == 1
                        targets[neg_mask] = 0.1 / n_entities
                        targets[pos_mask] = 1.0 - 0.1

                        loss = F.binary_cross_entropy_with_logits(scores, targets)
                else:
                    # CPU logic (same approach)
                    scores = model(subject, relation)
                    n_entities = scores.size(1)
                    targets = torch.zeros_like(scores, device=device)
                    for i in range(scores.size(0)):
                        targets[i][filter_o[i]] = 1

                    neg_mask = targets == 0
                    pos_mask = targets == 1
                    targets[neg_mask] = 0.1 / n_entities
                    targets[pos_mask] = 1.0 - 0.1

                    loss = F.binary_cross_entropy_with_logits(scores, targets)

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item()
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{(total_loss / (batch_idx + 1)):.4f}'})

        if device == "cuda":
            torch.cuda.empty_cache()

        # Evaluate
        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                metrics = evaluate_2pass(model, valid_dataloader, device)

            print("\nValidation:")
            print(f"  Loss:  {metrics['loss']:.4f}")
            print(f"  MR:    {metrics['mr']:.1f}")
            print(f"  MRR:   {metrics['mrr']:.4f}")
            print(f"  Hits@1: {metrics['hits@1']:.4f}")
            print(f"  Hits@3: {metrics['hits@3']:.4f}")
            print(f"  Hits@10:{metrics['hits@10']:.4f}")

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

            # Save best model
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mrr': best_mrr,
                }, f"{save_path}/best_model.pt")

    return model