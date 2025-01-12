import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict

class KGDataset(Dataset):
    def __init__(
        self,
        triples: List[Tuple[str, str, str]],
        entity2id: Dict[str, int],
        relation2id: Dict[str, int],
        negative_sample_size: int = 0
    ):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.negative_sample_size = negative_sample_size
        
        # Convert string triples to tensor of indices
        self.triple_indices = [
            (entity2id[h], relation2id[r], entity2id[t]) 
            for h, r, t in triples
        ]
        
        # Create filters for both directions
        self.filters_o = defaultdict(set)  # (head, rel) -> set of valid tails
        self.filters_s = defaultdict(set)  # (rel, tail) -> set of valid heads
        for h, r, t in self.triple_indices:
            self.filters_o[(h, r)].add(t)  # Forward direction
            self.filters_s[(r, t)].add(h)  # Reverse direction

    def __len__(self):
        return len(self.triple_indices)

    def __getitem__(self, idx):
        h, r, t = self.triple_indices[idx]
        return {
            'subject': h,
            'relation': r,
            'object': t,
            'filter_o': torch.LongTensor(list(self.filters_o[(h, r)])),
            'filter_s': torch.LongTensor(list(self.filters_s[(r, t)]))
        }

class KGDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.entity2id = {}
        self.relation2id = {}
        
    def load_data(self) -> Dict[str, Dataset]:
        """Load train/valid/test datasets."""
        # Load all triples
        train_triples = self._load_triples('train.txt')
        valid_triples = self._load_triples('valid.txt')
        test_triples = self._load_triples('test.txt')
        
        # Build vocabularies
        self._build_vocab(train_triples + valid_triples + test_triples)
        
        # Create datasets
        return {
            'train': KGDataset(train_triples, self.entity2id, self.relation2id),
            'valid': KGDataset(valid_triples, self.entity2id, self.relation2id),
            'test': KGDataset(test_triples, self.entity2id, self.relation2id)
        }
    
    def _load_triples(self, filename: str) -> List[Tuple[str, str, str]]:
        """Load triples from a tab-separated file."""
        triples = []
        filepath = self.data_dir / filename
        with open(filepath) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((h, r, t))
        return triples
    
    def _build_vocab(self, triples: List[Tuple[str, str, str]]):
        """Build entity and relation vocabularies."""
        entities = sorted(set(h for h,_,_ in triples) | set(t for _,_,t in triples))
        relations = sorted(set(r for _,r,_ in triples))
        
        self.entity2id = {ent: idx for idx, ent in enumerate(entities)}
        self.relation2id = {rel: idx for idx, rel in enumerate(relations)}

def create_dataloader(
    dataset: KGDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn
    )

def _collate_fn(batch):
    """Custom collate function to handle variable-sized filter lists."""
    # Convert indices to tensors here
    subject = torch.tensor([item['subject'] for item in batch], dtype=torch.long)
    relation = torch.tensor([item['relation'] for item in batch], dtype=torch.long)
    object_ = torch.tensor([item['object'] for item in batch], dtype=torch.long)
    
    # Handle both filter types
    max_filter_o_size = max(item['filter_o'].size(0) for item in batch)
    max_filter_s_size = max(item['filter_s'].size(0) for item in batch)
    
    filter_o = torch.zeros(len(batch), max_filter_o_size, dtype=torch.long)
    filter_s = torch.zeros(len(batch), max_filter_s_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        size_o = item['filter_o'].size(0)
        size_s = item['filter_s'].size(0)
        filter_o[i, :size_o] = item['filter_o']
        filter_s[i, :size_s] = item['filter_s']
    
    return {
        'subject': subject,
        'relation': relation,
        'object': object_,
        'filter_o': filter_o,
        'filter_s': filter_s
    }