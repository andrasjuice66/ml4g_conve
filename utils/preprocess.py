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
        relation2id: Dict[str, int]
    ):
        self.entity2id = entity2id
        self.relation2id = relation2id
        
        # Convert string triples to (idx_h, idx_r, idx_t)
        self.triple_indices = [
            (entity2id[h], relation2id[r], entity2id[t]) 
            for h, r, t in triples
        ]
        
        # Build filter dictionary for each (h, r)
        # so we can filter out all known positives when ranking
        self.filters_o = defaultdict(set)
        for h, r, t in self.triple_indices:
            self.filters_o[(h, r)].add(t)

    def __len__(self):
        return len(self.triple_indices)

    def __getitem__(self, idx):
        h, r, t = self.triple_indices[idx]
        return {
            'subject': h,
            'relation': r,
            'object': t,
            'filter_o': torch.LongTensor(list(self.filters_o[(h, r)]))
        }

class KGDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.entity2id = {}
        self.relation2id = {}

    def load_data(self) -> Dict[str, KGDataset]:
        # First load all triples
        train_triples = self._load_triples('train.txt')
        valid_triples = self._load_triples('valid.txt')
        test_triples = self._load_triples('test.txt')

        # Get entities and relations from training set only
        train_entities = set(h for h, _, _ in train_triples) | set(t for _, _, t in train_triples)
        train_relations = set(r for _, r, _ in train_triples)

        print(f"Original statistics:")
        print(f"Train: {len(train_triples)} triples")
        print(f"Valid: {len(valid_triples)} triples")
        print(f"Test: {len(test_triples)} triples")

        # Filter validation and test triples to only include training entities and relations
        valid_triples = [
            (h, r, t) for h, r, t in valid_triples 
            if h in train_entities and t in train_entities and r in train_relations
        ]
        
        test_triples = [
            (h, r, t) for h, r, t in test_triples 
            if h in train_entities and t in train_entities and r in train_relations
        ]

        print(f"\nAfter filtering (only training entities/relations):")
        print(f"Train: {len(train_triples)} triples")
        print(f"Valid: {len(valid_triples)} triples")
        print(f"Test: {len(test_triples)} triples")

        # Build vocab using only training entities and relations
        self._build_vocab_from_training(train_entities, train_relations)

        # Create dataset objects
        return {
            'train': KGDataset(train_triples, self.entity2id, self.relation2id),
            'valid': KGDataset(valid_triples, self.entity2id, self.relation2id),
            'test': KGDataset(test_triples, self.entity2id, self.relation2id),
        }

    def _build_vocab_from_training(self, train_entities: Set[str], train_relations: Set[str]):
        """Create entity2id and relation2id from training set only."""
        # Sort for deterministic ordering
        entities = sorted(train_entities)
        relations = sorted(train_relations)

        # Create mappings
        self.entity2id = {ent: idx for idx, ent in enumerate(entities)}
        self.relation2id = {rel: idx for idx, rel in enumerate(relations)}

        print(f"\nVocabulary statistics:")
        print(f"Number of entities: {len(self.entity2id)}")
        print(f"Number of relations: {len(self.relation2id)}")

    def _load_triples(self, filename: str) -> List[Tuple[str, str, str]]:
        """Load triples from file."""
        triples = []
        filepath = self.data_dir / filename
        with open(filepath) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((h, r, t))
        return triples



def _collate_fn(batch):
    subject = torch.tensor([item['subject'] for item in batch], dtype=torch.long)
    relation = torch.tensor([item['relation'] for item in batch], dtype=torch.long)
    object_ = torch.tensor([item['object'] for item in batch], dtype=torch.long)
    
    max_filter_size = max(item['filter_o'].size(0) for item in batch)
    filter_o = torch.zeros(len(batch), max_filter_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        size_o = item['filter_o'].size(0)
        filter_o[i, :size_o] = item['filter_o']

    return {
        'subject': subject,
        'relation': relation,
        'object': object_,
        'filter_o': filter_o
    }

def create_dataloader(dataset: KGDataset, batch_size: int, shuffle=True) -> DataLoader:
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=_collate_fn,
        pin_memory=True, 
        num_workers=2,   
            persistent_workers=True  
        )