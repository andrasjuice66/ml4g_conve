# utils/preprocess.py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
from collections import defaultdict

class KGDataset(Dataset):
    """
    A knowledge graph dataset that stores all triples in index form and
    maintains sets of valid tails/heads for (h, r) or (r, t) to do filtered evaluation
    and multi-label training targets.
    """
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

        # Build filter dictionary for each (h, r) so we can mark known tails as positives
        self.filters_o = defaultdict(set)
        # Build filter dictionary for each (r, t) so we can mark known heads as positives
        self.filters_h = defaultdict(set)
        for h, r, t in self.triple_indices:
            self.filters_o[(h, r)].add(t)
            self.filters_h[(r, t)].add(h)

    def __len__(self):
        return len(self.triple_indices)

    def __getitem__(self, idx):
        h, r, t = self.triple_indices[idx]
        return {
            'subject': h,
            'relation': r,
            'object': t,
            'filter_o': torch.LongTensor(list(self.filters_o[(h, r)])),
            'filter_h': torch.LongTensor(list(self.filters_h[(r, t)]))
        }

class KGDataLoader:
    """
    A loader that reads train/valid/test .txt files and
    splits them into custom Dataset objects.
    """
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.entity2id = {}
        self.relation2id = {}

    def load_data(self) -> Dict[str, KGDataset]:
        # Load all triples from train/valid/test
        train_triples = self._load_triples('train.txt')
        valid_triples = self._load_triples('valid.txt')
        test_triples = self._load_triples('test.txt')

        # Gather entities/relations from training set only
        train_entities = set(h for h, _, _ in train_triples) | set(t for _, _, t in train_triples)
        train_relations = set(r for _, r, _ in train_triples)

        print(f"Original statistics:")
        print(f"Train: {len(train_triples)} triples")
        print(f"Valid: {len(valid_triples)} triples")
        print(f"Test: {len(test_triples)} triples")

        # Filter valid/test so that only in-vocab entities/relations remain
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

        # Build vocab from training set
        self._build_vocab_from_training(train_entities, train_relations)

        # Create dataset objects
        return {
            'train': KGDataset(train_triples, self.entity2id, self.relation2id),
            'valid': KGDataset(valid_triples, self.entity2id, self.relation2id),
            'test': KGDataset(test_triples, self.entity2id, self.relation2id),
        }

    def _build_vocab_from_training(self, train_entities: Set[str], train_relations: Set[str]):
        """Initialize entity2id and relation2id from training."""
        entities = sorted(train_entities)
        relations = sorted(train_relations)

        self.entity2id = {ent: idx for idx, ent in enumerate(entities)}
        self.relation2id = {rel: idx for idx, rel in enumerate(relations)}

        print(f"\nVocabulary statistics:")
        print(f"Number of entities: {len(self.entity2id)}")
        print(f"Number of relations: {len(self.relation2id)}")

    def _load_triples(self, filename: str) -> List[Tuple[str, str, str]]:
        """Loads each line of the .txt file into (head, relation, tail)."""
        triples = []
        filepath = self.data_dir / filename
        with open(filepath, 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((h, r, t))
        return triples

def _collate_fn(batch):
    """
    Merges a list of samples to form a mini-batch for DataLoader.
    Includes 'filter_o' for all valid tails, 'filter_h' for all valid heads.
    """
    subject = torch.tensor([item['subject'] for item in batch], dtype=torch.long)
    relation = torch.tensor([item['relation'] for item in batch], dtype=torch.long)
    object_ = torch.tensor([item['object'] for item in batch], dtype=torch.long)

    # Pad filter_o so each row has same size
    max_filter_size_o = max(item['filter_o'].size(0) for item in batch)
    filter_o = torch.zeros(len(batch), max_filter_size_o, dtype=torch.long)
    for i, item in enumerate(batch):
        size_o = item['filter_o'].size(0)
        filter_o[i, :size_o] = item['filter_o']

    # Pad filter_h similarly
    max_filter_size_h = max(item['filter_h'].size(0) for item in batch)
    filter_h = torch.zeros(len(batch), max_filter_size_h, dtype=torch.long)
    for i, item in enumerate(batch):
        size_h = item['filter_h'].size(0)
        filter_h[i, :size_h] = item['filter_h']

    return {
        'subject': subject,
        'relation': relation,
        'object': object_,
        'filter_o': filter_o,
        'filter_h': filter_h
    }

def create_dataloader(dataset: KGDataset, batch_size: int, shuffle=True) -> DataLoader:
    """Wraps the KGDataset in a DataLoader with custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True
    )