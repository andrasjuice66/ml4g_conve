import torch
import numpy as np
import logging
import wandb

log = logging.getLogger(__name__)


def train(model, config, train_loader, num_entities, valid_loader, optimizer, device):
    """
    Trains the model using the provided configuration and data loaders.

    Args:
        model: The neural network model to train
        config: Dictionary containing training configuration parameters
        train_loader: DataLoader for training data
        num_entities: Total number of entities in the knowledge graph
        valid_loader: DataLoader for validation data
        optimizer: PyTorch optimizer instance
        device: Device to run the training on (cuda/cpu)
    """
    for epoch in range(config['epochs']):
            model.train()
            total_loss = 0.0

            for i, str2var in enumerate(train_loader):
                head = str2var['head'].to(device)
                rel = str2var['relation'].to(device)
                valid_tails = str2var['valid_tails'].to(device)

                B = head.shape[0]

                target = torch.zeros((B, num_entities), dtype=torch.float, device=device)
                for row_idx in range(B):
                    valid_ids = valid_tails[row_idx][valid_tails[row_idx] != -1]
                    target[row_idx, valid_ids] = 1.0

                target = ((1.0 - config['label_smoothing']) * target) + (config['label_smoothing'] / num_entities)

                pred = model.forward(head, rel)
                loss = model.loss(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                train_loader.state = {'loss': loss.item()}

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{config['epochs']}  -  Avg Train Loss: {avg_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                print("Evaluating on the validation set...")
                evaluation(model, valid_loader)

   


def evaluation(model, eval_loader):
    """
    Computes metrics (e.g., Hits@k, mean rank, MRR) using filtered rankings,
    but keeps everything on GPU to reduce overhead.

    Args:
        model: The neural network model to evaluate
        eval_loader: DataLoader containing evaluation data

    Returns:
        tuple: (metrics, wandb_metrics)
            - metrics: Dictionary containing all computed metrics
            - wandb_metrics: Filtered subset of metrics for wandb logging
    """

    device = next(model.parameters()).device
    model.eval()

    hits_left = [[] for _ in range(10)]
    hits_right = [[] for _ in range(10)]
    hits = [[] for _ in range(10)]
    ranks = []
    ranks_left = []
    ranks_right = []

    with torch.no_grad():
        for i, str2var in enumerate(eval_loader):
            head = str2var['head'].to(device)
            tail = str2var['tail'].to(device)
            rel = str2var['relation'].to(device)
            rel_rev = str2var['reverse_relation'].to(device)
            valid_tails = str2var['valid_tails'].to(device)
            valid_heads = str2var['valid_heads'].to(device)

            pred1 = model.forward(head, rel)
            pred2 = model.forward(tail, rel_rev)

            B = head.size(0)
            for j in range(B):
                filter1 = valid_tails[j]
                filter1 = filter1[filter1 != -1]
                target_entity = tail[j]

                correct_score_1 = pred1[j, target_entity].clone()
                pred1[j, filter1] = -1e6
                pred1[j, target_entity] = correct_score_1

                filter2 = valid_heads[j]
                filter2 = filter2[filter2 != -1]
                correct_score_2 = pred2[j, head[j]].clone()
                pred2[j, filter2] = -1e6
                pred2[j, head[j]] = correct_score_2

            sorted1 = torch.argsort(pred1, dim=1, descending=True)
            sorted2 = torch.argsort(pred2, dim=1, descending=True)
            
            for j in range(B):
                rank1_idx = torch.nonzero(sorted1[j] == tail[j], as_tuple=True)[0]
                rank1 = rank1_idx.item()

                rank2_idx = torch.nonzero(sorted2[j] == head[j], as_tuple=True)[0]
                rank2 = rank2_idx.item()

                ranks_left.append(rank1 + 1)
                ranks_right.append(rank2 + 1)
                ranks.append(rank1 + 1)
                ranks.append(rank2 + 1)

                for hits_level in range(10):
                    if rank1 <= hits_level:
                        hits_left[hits_level].append(1.0)
                    else:
                        hits_left[hits_level].append(0.0)

                    if rank2 <= hits_level:
                        hits_right[hits_level].append(1.0)
                    else:
                        hits_right[hits_level].append(0.0)

                    if rank1 <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                    if rank2 <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

            eval_loader.state = {'loss': 0}

    ranks_left_arr = np.array(ranks_left)
    ranks_right_arr = np.array(ranks_right)
    ranks_arr = np.array(ranks)

    metrics = {
        'mean_rank_left': np.mean(ranks_left_arr),
        'mean_rank_right': np.mean(ranks_right_arr),
        'mean_rank': np.mean(ranks_arr),
        'mrr_left': np.mean(1./ranks_left_arr),
        'mrr_right': np.mean(1./ranks_right_arr),
        'mrr': np.mean(1./ranks_arr)
    }

    for i in range(10):
        metrics[f'hits_left@{i+1}'] = np.mean(hits_left[i])
        metrics[f'hits_right@{i+1}'] = np.mean(hits_right[i])
        metrics[f'hits@{i+1}'] = np.mean(hits[i])

    wandb_metrics = {
        'mean_rank': metrics['mean_rank'],
        'mrr': metrics['mrr']
    }

    for i in range(10):
        wandb_metrics[f'hits@{i+1}'] = metrics[f'hits@{i+1}']

    wandb.log(wandb_metrics)

    print('')
    for i in range(10):
        print(f'Hits left @{i+1}: {metrics[f"hits_left@{i+1}"]}')
        print(f'Hits right @{i+1}: {metrics[f"hits_right@{i+1}"]}')
        print(f'Hits @{i+1}: {metrics[f"hits@{i+1}"]}')

    print(f'Mean rank left: {metrics["mean_rank_left"]}')
    print(f'Mean rank right: {metrics["mean_rank_right"]}')
    print(f'Mean rank: {metrics["mean_rank"]}')

    print(f'Mean reciprocal rank left: {metrics["mrr_left"]}')
    print(f'Mean reciprocal rank right: {metrics["mrr_right"]}')
    print(f'Mean reciprocal rank: {metrics["mrr"]}')

    return metrics, wandb_metrics
