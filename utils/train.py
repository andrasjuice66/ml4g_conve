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
    For each sample, we do:
      1) Tail mode: fix (h, r), predict t
      2) Head mode: fix (r, t), predict h
    Then compute combined ranks/hits.
    This can be closer to the standard "raw/filtered" link prediction metrics in KGE research.
    """
    model.eval()
    ranks_all = []
    hits_all = {1: 0, 3: 0, 10: 0}
    total_loss = 0

    def update_ranks(filtered_scores, gold_idx, ranks_list, hits_dict):
        gold_score = filtered_scores[gold_idx]
        rank = (filtered_scores >= gold_score).sum().item()
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

        # ====== TAIL PREDICTION ======
        scores_tail = model.forward(subject, relation)
        n_entities = scores_tail.size(1)

        # Multi-hot target for tail
        targets_tail = torch.zeros_like(scores_tail, device=device)
        for i in range(scores_tail.size(0)):
            targets_tail[i][filter_o[i]] = 1
        neg_mask_tail = (targets_tail == 0)
        pos_mask_tail = (targets_tail == 1)

        targets_tail[neg_mask_tail] = 0.1 / n_entities
        targets_tail[pos_mask_tail] = 1.0 - 0.1
        loss_tail = F.binary_cross_entropy_with_logits(scores_tail, targets_tail)

        # Filter for ranking
        scores_tail = torch.sigmoid(scores_tail)
        for i in range(scores_tail.size(0)):
            mask = torch.ones(n_entities, dtype=torch.bool, device=device)
            mask[filter_o[i]] = False
            mask[object_true[i]] = True
            filtered_scores = scores_tail[i].clone()
            filtered_scores[~mask] = float('-inf')
            # Update tail rank
            update_ranks(filtered_scores, object_true[i].item(), ranks_all, hits_all)

        # ====== HEAD PREDICTION ======
        scores_head = model.forward_head(object_true, relation)
        targets_head = torch.zeros_like(scores_head, device=device)
        for i in range(scores_head.size(0)):
            targets_head[i][filter_h[i]] = 1
        neg_mask_head = (targets_head == 0)
        pos_mask_head = (targets_head == 1)

        targets_head[neg_mask_head] = 0.1 / n_entities
        targets_head[pos_mask_head] = 1.0 - 0.1
        loss_head = F.binary_cross_entropy_with_logits(scores_head, targets_head)

        scores_head = torch.sigmoid(scores_head)
        for i in range(scores_head.size(0)):
            mask = torch.ones(n_entities, dtype=torch.bool, device=device)
            mask[filter_h[i]] = False
            mask[subject[i]] = True
            filtered_scores = scores_head[i].clone()
            filtered_scores[~mask] = float('-inf')
            # Update head rank
            update_ranks(filtered_scores, subject[i].item(), ranks_all, hits_all)

        # Average of tail+head loss
        total_loss += 0.5 * (loss_tail.item() + loss_head.item())

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