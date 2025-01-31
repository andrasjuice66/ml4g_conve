import torch
import json
from torch.utils.data import Dataset


class SimpleVocab:
    def __init__(self, token2id):
        self.token2id = token2id
        self.num_token = len(token2id)

def build_vocab(entity2id_path, relation2id_path):
    with open(entity2id_path, 'r') as f:
        entity2id = json.load(f)
    with open(relation2id_path, 'r') as f:
        relation2id = json.load(f)

    vocab = {
        'e1': SimpleVocab(entity2id),
        'rel': SimpleVocab(relation2id)
    }
    return vocab

class JSONLinkPredictionDataset(Dataset):
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
            'e1': row['e1'],
            'rel': row['rel'],
            'rel_eval': row['rel_eval'],
            'e2': row['e2'],  # might be -1 in training
            'e2_multi1': row['e2_multi1'],
            'e2_multi2': row['e2_multi2']
        }

def collate_fn(batch):
    e1_list       = [x['e1'] for x in batch]
    rel_list      = [x['rel'] for x in batch]
    rel_eval_list = [x['rel_eval'] for x in batch]
    e2_list       = [x['e2'] for x in batch]
    e2m1_list     = [x['e2_multi1'] for x in batch]
    e2m2_list     = [x['e2_multi2'] for x in batch]

    e1_tensor       = torch.tensor(e1_list,       dtype=torch.long)
    rel_tensor      = torch.tensor(rel_list,      dtype=torch.long)
    rel_eval_tensor = torch.tensor(rel_eval_list, dtype=torch.long)
    e2_tensor       = torch.tensor(e2_list,       dtype=torch.long)

    max_len_m1 = max(len(row) for row in e2m1_list) if e2m1_list else 0
    max_len_m2 = max(len(row) for row in e2m2_list) if e2m2_list else 0
    if max_len_m1 == 0: max_len_m1 = 1
    if max_len_m2 == 0: max_len_m2 = 1

    B = len(batch)
    e2m1_tensor = torch.full((B, max_len_m1), -1, dtype=torch.long)
    e2m2_tensor = torch.full((B, max_len_m2), -1, dtype=torch.long)

    for i in range(B):
        e2m1 = e2m1_list[i]
        e2m2 = e2m2_list[i]
        e2m1_tensor[i, :len(e2m1)] = torch.tensor(e2m1, dtype=torch.long)
        e2m2_tensor[i, :len(e2m2)] = torch.tensor(e2m2, dtype=torch.long)

    return {
        'e1': e1_tensor,
        'rel': rel_tensor,
        'rel_eval': rel_eval_tensor,
        'e2': e2_tensor,
        'e2_multi1': e2m1_tensor,
        'e2_multi2': e2m2_tensor,
    }
