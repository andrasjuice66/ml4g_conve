# models/conve.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvE(nn.Module):
    """
    The ConvE model with forward passes for tail-prediction and head-prediction.
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,
        use_stacked_embeddings: bool = True,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        feature_map_dropout: float = 0.2,
        use_bias: bool = True
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.embedding_shape1 = embedding_shape1
        self.embedding_shape2 = embedding_dim // embedding_shape1
        self.use_stacked_embeddings = use_stacked_embeddings

        # Embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Dropout layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.feature_map_dropout = nn.Dropout(feature_map_dropout)

        # Convolution layer
        self.conv2d = nn.Conv2d(1, 32, (3, 3), padding=0, bias=use_bias)

        # Calculate the size after convolution
        if use_stacked_embeddings:
            conv_output_height = 2 * self.embedding_shape1 - 2  # Adjusted for padding=0
        else:
            conv_output_height = 2 * self.embedding_shape1 - 2  # Adjusted for padding=0

        conv_output_width = self.embedding_shape2 - 2  # Adjusted for padding=0

        # Output layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        fc_length = 32 * conv_output_height * conv_output_width
        self.fc = nn.Linear(fc_length, embedding_dim, bias=use_bias)

        # Initialize embeddings with smaller values
        nn.init.xavier_normal_(self.entity_embeddings.weight, gain=0.1)
        nn.init.xavier_normal_(self.relation_embeddings.weight, gain=0.1)
        
        # Initialize conv layers properly
        nn.init.xavier_normal_(self.conv2d.weight, gain=1.414)
        nn.init.zeros_(self.conv2d.bias)

        # Add bias term for final scoring
        self.bias = nn.Parameter(torch.zeros(num_entities))

    def _reshape_embeddings(self, e1_embedded, rel_embedded):
        """
        Stack or interleave subject and relation embeddings to feed into the 2D conv layer.
        """
        if self.use_stacked_embeddings:
            # Original stacked version: just stack (subject, relation) on vertical dimension
            stacked_inputs = torch.stack([e1_embedded, rel_embedded], dim=1)
            stacked_inputs = stacked_inputs.reshape(-1, 1, 2 * self.embedding_shape1, self.embedding_shape2)
        else:
            # Interleaved version
            e1 = e1_embedded.view(-1, self.embedding_shape1, self.embedding_shape2)
            rel = rel_embedded.view(-1, self.embedding_shape1, self.embedding_shape2)
            interleaved = torch.zeros_like(e1.repeat(1, 2, 1))
            interleaved[:, 0::2, :] = e1
            interleaved[:, 1::2, :] = rel
            stacked_inputs = interleaved.unsqueeze(1)
        return stacked_inputs

    def forward(self, subject_idx, relation_idx):
        """
        Predict tail entities: returns a score vector of shape (batch_size, num_entities).
        """
        batch_size = subject_idx.size(0)

        # Get embeddings
        subject_embedded = self.entity_embeddings(subject_idx)
        relation_embedded = self.relation_embeddings(relation_idx)

        # Reshape and dropout
        stacked_inputs = self._reshape_embeddings(subject_embedded, relation_embedded)
        stacked_inputs = self.input_dropout(stacked_inputs)

        # Convolution
        x = self.conv2d(stacked_inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        # Flatten + FC
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Final scoring against all entities
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        x = x + self.bias  # Add bias before sigmoid
        return x

    def forward_head(self, tail_idx, relation_idx):
        """
        Predict head entities:
        Same architecture, but we input (tail, relation) and try to predict the subject.
        """
        batch_size = tail_idx.size(0)

        # Embeddings
        tail_embedded = self.entity_embeddings(tail_idx)
        relation_embedded = self.relation_embeddings(relation_idx)

        # Reshape
        stacked_inputs = self._reshape_embeddings(tail_embedded, relation_embedded)
        stacked_inputs = self.input_dropout(stacked_inputs)

        # Convolution
        x = self.conv2d(stacked_inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        # Flatten + FC
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Final scoring against all entities
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        x = x + self.bias  # Add bias before sigmoid
        return x