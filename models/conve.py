import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvE(nn.Module):
    def __init__(
        self, 
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,  # for 2D reshaping
        hidden_size: int = 9728,  # 32 * 19 * 16
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        feature_map_dropout: float = 0.2,
        use_bias: bool = True
    ):
        super().__init__()
        
        # Calculate embedding shape2 based on embedding_dim and shape1
        self.embedding_shape2 = embedding_dim // embedding_shape1
        self.embedding_dim = embedding_dim
        self.embedding_shape1 = embedding_shape1
        
        # Entity and relation embeddings with padding_idx=0
        self.emb_e = nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        
        # Dropout layers
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)
        
        # Convolution layer with exact settings from paper
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)
        
        # Batch normalization layers
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
        # Learnable bias for each entity
        self.b = nn.Parameter(torch.zeros(num_entities))
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, embedding_dim)
        
        # Initialize embeddings
        self.init()

    def init(self):
        """Initialize embeddings like the original implementation."""
        nn.init.xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        # Get and reshape embeddings
        e1_embedded = self.emb_e(e1).view(-1, 1, self.embedding_shape1, self.embedding_shape2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.embedding_shape1, self.embedding_shape2)
        
        # Stack inputs along dimension 2 (height)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
        
        # First batch norm
        stacked_inputs = self.bn0(stacked_inputs)
        
        # Input dropout
        x = self.inp_drop(stacked_inputs)
        
        # Convolution
        x = self.conv1(x)
        
        # Batch norm and activation
        x = self.bn1(x)
        x = F.relu(x)
        
        # Feature map dropout
        x = self.feature_map_drop(x)
        
        # Reshape for fully connected layer
        x = x.view(x.shape[0], -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        # Hidden dropout
        x = self.hidden_drop(x)
        
        # Final batch norm and activation
        x = self.bn2(x)
        x = F.relu(x)
        
        # Score against all entities
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        
        # Add bias term
        x += self.b
        
        # Final sigmoid
        #pred = torch.sigmoid(x)
        
        return x
    
    def forward_head(self, e2, rel):
        """
        A 'head-mode' forward pass: predicts the score of each possible 'head'
        given (relation=rel, object=e2).
        We can do this simply by re-using our 'forward' code, but swapping
        what we call subject vs. object.
        """
        # In practice, we do the same steps but we feed e2 into self.emb_e
        # as if it were e1. So essentially:
        return self.forward(e2, rel)

