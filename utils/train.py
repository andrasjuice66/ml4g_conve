import torch
import numpy as np
import logging
import wandb

log = logging.getLogger(__name__)


def train(model, config, train_loader, num_entities, valid_loader, optimizer, device):
    for epoch in range(config['epochs']):
            model.train()
            total_loss = 0.0

            for i, str2var in enumerate(train_loader):
                e1 = str2var['e1'].to(device)
                rel = str2var['rel'].to(device)
                e2_multi1 = str2var['e2_multi1'].to(device)

                B = e1.shape[0]

                # Build a multi-hot target from e2_multi1
                target = torch.zeros((B, num_entities), dtype=torch.float, device=device)
                for row_idx in range(B):
                    valid_ids = e2_multi1[row_idx][e2_multi1[row_idx] != -1]
                    target[row_idx, valid_ids] = 1.0

                # Label smoothing if needed
                target = ((1.0 - config['label_smoothing']) * target) + (config['label_smoothing'] / num_entities)

                pred = model.forward(e1, rel)
                loss = model.loss(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # track loader state
                train_loader.state = {}
                train_loader.state['loss'] = loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{config['epochs']}  -  Avg Train Loss: {avg_loss:.4f}")
            # 6) Evaluate periodically
            if (epoch + 1) % 5 == 0:
                print("Evaluating on the validation set...")
                evaluation(model, valid_loader)

   


def evaluation(model, eval_loader):
    """
    Computes metrics (e.g., Hits@k, mean rank, MRR) using filtered rankings,
    but keeps everything on GPU to reduce overhead.

    If memory usage is still too large, try reducing --test-batch-size.
    """

    device = next(model.parameters()).device
    model.eval()

    # We'll keep track of ranks in Python lists.
    # Or optionally, you could keep them in GPU tensors and .cpu() them once at the end.
    hits_left = [[] for _ in range(10)]
    hits_right = [[] for _ in range(10)]
    hits = [[] for _ in range(10)]
    ranks = []
    ranks_left = []
    ranks_right = []

    # Turn off gradient calculations for eval
    with torch.no_grad():
        for i, str2var in enumerate(eval_loader):
            # Move everything to GPU
            e1 = str2var['e1'].to(device)
            e2 = str2var['e2'].to(device)
            rel = str2var['rel'].to(device)
            rel_eval = str2var['rel_eval'].to(device)
            e2_multi1 = str2var['e2_multi1'].to(device)
            e2_multi2 = str2var['e2_multi2'].to(device)

            # Forward passes
            pred1 = model.forward(e1, rel)         # shape [B, num_entities]
            pred2 = model.forward(e2, rel_eval)    # shape [B, num_entities]

            B = e1.size(0)
            # Filter and re-instate correct answer scores
            for j in range(B):
                # Known correct answers for the (e1, rel, ???) are in e2_multi1
                # We'll "filter" them by setting their score to a large negative
                filter1 = e2_multi1[j]
                filter1 = filter1[filter1 != -1]  # remove padding
                target_entity = e2[j]  # correct answer for (e1, rel)

                # Save the correct score, set filter to -1e6
                correct_score_1 = pred1[j, target_entity].clone()
                pred1[j, filter1] = -1e6
                # restore the target
                pred1[j, target_entity] = correct_score_1

                # For the inverse side: (e2, rel_eval, ???)
                filter2 = e2_multi2[j]
                filter2 = filter2[filter2 != -1]
                correct_score_2 = pred2[j, e1[j]].clone()
                pred2[j, filter2] = -1e6
                pred2[j, e1[j]] = correct_score_2

            # Now we do the ranking on GPU
            # argsort each row in descending order => shape [B, num_entities]
            sorted1 = torch.argsort(pred1, dim=1, descending=True)
            sorted2 = torch.argsort(pred2, dim=1, descending=True)
            
            for j in range(B):
                # rank of e2[j] in sorted1[j]
                # This returns a 1D tensor of all indices where sorted1[j] == e2[j]
                # which should be exactly one element.
                rank1_idx = torch.nonzero(sorted1[j] == e2[j], as_tuple=True)[0]
                rank1 = rank1_idx.item()  # integer rank index

                # rank of e1[j] in sorted2[j]
                rank2_idx = torch.nonzero(sorted2[j] == e1[j], as_tuple=True)[0]
                rank2 = rank2_idx.item()

                # Convert to 1-based rank
                ranks_left.append(rank1 + 1)
                ranks_right.append(rank2 + 1)
                ranks.append(rank1 + 1)
                ranks.append(rank2 + 1)

                # Hits@k for k in [1..10]
                for hits_level in range(10):
                    if rank1 <= hits_level:
                        hits_left[hits_level].append(1.0)
                    else:
                        hits_left[hits_level].append(0.0)

                    if rank2 <= hits_level:
                        hits_right[hits_level].append(1.0)
                    else:
                        hits_right[hits_level].append(0.0)

                    # overall hits
                    if rank1 <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                    if rank2 <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

            # If your eval loader tracks a loss or progress bar:
            eval_loader.state = {}
            eval_loader.state['loss'] = 0

    # We have gathered all ranks in Python lists. We can print stats now.
    # Convert them to NumPy for easier mean computations
    ranks_left_arr = np.array(ranks_left)
    ranks_right_arr = np.array(ranks_right)
    ranks_arr = np.array(ranks)

    # After computing all the metrics
    metrics = {
        'mean_rank_left': np.mean(ranks_left_arr),
        'mean_rank_right': np.mean(ranks_right_arr),
        'mean_rank': np.mean(ranks_arr),
        'mrr_left': np.mean(1./ranks_left_arr),
        'mrr_right': np.mean(1./ranks_right_arr),
        'mrr': np.mean(1./ranks_arr)
    }

    # Add Hits@k metrics
    for i in range(10):
        metrics[f'hits_left@{i+1}'] = np.mean(hits_left[i])
        metrics[f'hits_right@{i+1}'] = np.mean(hits_right[i])
        metrics[f'hits@{i+1}'] = np.mean(hits[i])

    # Log metrics to wandb
    wandb.log(metrics)

    # Still log to console as before
    log.info('')
    for i in range(10):
        log.info('Hits left @{0}: {1}', i+1, metrics[f'hits_left@{i+1}'])
        log.info('Hits right @{0}: {1}', i+1, metrics[f'hits_right@{i+1}'])
        log.info('Hits @{0}: {1}', i+1, metrics[f'hits@{i+1}'])

    log.info('Mean rank left: {0}', metrics['mean_rank_left'])
    log.info('Mean rank right: {0}', metrics['mean_rank_right'])
    log.info('Mean rank: {0}', metrics['mean_rank'])

    log.info('Mean reciprocal rank left: {0}', metrics['mrr_left'])
    log.info('Mean reciprocal rank right: {0}', metrics['mrr_right'])
    log.info('Mean reciprocal rank: {0}', metrics['mrr'])

    return metrics  # Return metrics dictionary for use elsewhere
