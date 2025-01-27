import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnConvE(nn.Module):
    """ConvE model with multi-head self-attention mechanism for knowledge graph link prediction.
    
    Extends the base ConvE model by adding a self-attention layer to capture dependencies
    between entity and relation embeddings before the convolution operation.
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,
        num_attention_heads: int = 8,
        use_stacked_embeddings: bool = True,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        feature_map_dropout: float = 0.2
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

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=input_dropout
        )

        # Dropout layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.feature_map_dropout = nn.Dropout(feature_map_dropout)
        self.attn_dropout = nn.Dropout(input_dropout)

        # Convolution layer
        self.conv2d = nn.Conv2d(1, 32, (3, 3), padding=1)

        # Calculate the size after convolution
        if use_stacked_embeddings:
            conv_output_height = 2 * self.embedding_shape1
        else:
            conv_output_height = 2 * self.embedding_shape1
            
        conv_output_width = self.embedding_shape2
        
        # Output layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        fc_length = 32 * conv_output_height * conv_output_width
        self.fc = nn.Linear(fc_length, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    def _reshape_embeddings(self, e1_embedded, rel_embedded):
        """Reshape embeddings after attention for convolution input."""
        if self.use_stacked_embeddings:
            stacked_inputs = torch.stack([e1_embedded, rel_embedded], dim=1)
            stacked_inputs = stacked_inputs.reshape(-1, 1, 2 * self.embedding_shape1, self.embedding_shape2)
        else:
            e1 = e1_embedded.view(-1, self.embedding_shape1, self.embedding_shape2)
            rel = rel_embedded.view(-1, self.embedding_shape1, self.embedding_shape2)
            
            interleaved = torch.zeros_like(e1.repeat(1, 2, 1))
            interleaved[:, 0::2, :] = e1
            interleaved[:, 1::2, :] = rel
            
            stacked_inputs = interleaved.unsqueeze(1)
        
        return stacked_inputs

    def forward(self, subject_idx, relation_idx):
        """Forward pass for tail prediction."""
        batch_size = subject_idx.size(0)
        
        # Get embeddings
        subject_embedded = self.entity_embeddings(subject_idx).view(-1, self.embedding_dim)
        relation_embedded = self.relation_embeddings(relation_idx).view(-1, self.embedding_dim)

        # Apply self-attention
        combined = torch.stack([subject_embedded, relation_embedded], dim=1)
        combined = combined.permute(1, 0, 2)
        attn_output, _ = self.self_attn(combined, combined, combined)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = self.attn_dropout(attn_output)

        # Extract attended embeddings
        subject_attended = attn_output[:, 0, :]
        relation_attended = attn_output[:, 1, :]

        # Reshape for convolution
        stacked_inputs = self._reshape_embeddings(subject_attended, relation_attended)
        
        # Apply input dropout
        stacked_inputs = self.input_dropout(stacked_inputs)
        
        # Convolution
        x = self.conv2d(stacked_inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)
        
        # Fully connected layers
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Score computation
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        
        return x

    def forward_head(self, tail_idx, relation_idx):
        """Forward pass for head prediction."""
        batch_size = tail_idx.size(0)
        
        # Get embeddings
        tail_embedded = self.entity_embeddings(tail_idx).view(-1, self.embedding_dim)
        relation_embedded = self.relation_embeddings(relation_idx).view(-1, self.embedding_dim)

        # Apply self-attention
        combined = torch.stack([tail_embedded, relation_embedded], dim=1)
        combined = combined.permute(1, 0, 2)
        attn_output, _ = self.self_attn(combined, combined, combined)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = self.attn_dropout(attn_output)

        # Extract attended embeddings
        tail_attended = attn_output[:, 0, :]
        relation_attended = attn_output[:, 1, :]

        # Reshape for convolution
        stacked_inputs = self._reshape_embeddings(tail_attended, relation_attended)
        
        # Rest of the forward pass is the same
        stacked_inputs = self.input_dropout(stacked_inputs)
        x = self.conv2d(stacked_inputs)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        
        return x