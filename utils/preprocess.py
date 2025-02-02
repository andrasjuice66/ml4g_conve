import torch
import json
from torch.utils.data import Dataset


class Vocab:
    def __init__(self, token2id):
        self.token2id = token2id
        self.num_token = len(token2id)

def build_vocab(entity2id_path, relation2id_path):
    with open(entity2id_path, 'r') as f:
        entity2id = json.load(f)
    with open(relation2id_path, 'r') as f:
        relation2id = json.load(f)

    vocab = {
        'head': Vocab(entity2id),
        'relation': Vocab(relation2id)
    }
    return vocab

class LinkPredictionDataset(Dataset):
    def __init__(self, json_path, mode='train'):
        super().__init__()
        self.mode = mode
        self.samples = []
        with open(json_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                if 'rel_eval' not in d:
                    d['rel_eval'] = d['rel']
                self.samples.append(d)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        return {
            'head': row['head'],
            'relation': row['relation'],
            'reverse_relation': row['reverse_relation'],
            'tail': row['tail'],
            'valid_tails': row['valid_tails'],
            'valid_heads': row['valid_heads']
        }

def collate_fn(batch):
    head_list       = [x['head'] for x in batch]
    rel_list        = [x['relation'] for x in batch]
    rel_rev_list    = [x['reverse_relation'] for x in batch]
    tail_list       = [x['tail'] for x in batch]
    valid_tails_list = [x['valid_tails'] for x in batch]
    valid_heads_list = [x['valid_heads'] for x in batch]

    head_tensor      = torch.tensor(head_list,       dtype=torch.long)
    rel_tensor       = torch.tensor(rel_list,        dtype=torch.long)
    rel_rev_tensor   = torch.tensor(rel_rev_list,    dtype=torch.long)
    tail_tensor      = torch.tensor(tail_list,       dtype=torch.long)

    max_len_tails = max(len(row) for row in valid_tails_list) if valid_tails_list else 0
    max_len_heads = max(len(row) for row in valid_heads_list) if valid_heads_list else 0
    if max_len_tails == 0: max_len_tails = 1
    if max_len_heads == 0: max_len_heads = 1

    B = len(batch)
    valid_tails_tensor = torch.full((B, max_len_tails), -1, dtype=torch.long)
    valid_heads_tensor = torch.full((B, max_len_heads), -1, dtype=torch.long)

    for i in range(B):
        valid_tails = valid_tails_list[i]
        valid_heads = valid_heads_list[i]
        valid_tails_tensor[i, :len(valid_tails)] = torch.tensor(valid_tails, dtype=torch.long)
        valid_heads_tensor[i, :len(valid_heads)] = torch.tensor(valid_heads, dtype=torch.long)

    return {
        'head': head_tensor,
        'relation': rel_tensor,
        'reverse_relation': rel_rev_tensor,
        'tail': tail_tensor,
        'valid_tails': valid_tails_tensor,
        'valid_heads': valid_heads_tensor,
    }
