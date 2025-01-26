import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEWithAttention(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,
        num_attention_heads: int = 8,
        hidden_size: int = 9728,
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

        # Entity and relation embeddings
        self.emb_e = nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, embedding_dim, padding_idx=0)

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=input_dropout
        )

        # Dropout layers
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)
        self.attn_drop = nn.Dropout(input_dropout)

        # Convolution layer
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)

        # Batch normalization layers
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # Learnable bias and FC layer
        self.b = nn.Parameter(torch.zeros(num_entities))
        self.fc = nn.Linear(hidden_size, embedding_dim)

        self.init()

    def init(self):
        """Initialize embeddings using Xavier normalization"""
        nn.init.xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        # Get raw embeddings
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)

        # ========== Self-Attention Branch ==========
        # Combine entity and relation embeddings
        combined = torch.stack([e1_emb, rel_emb], dim=1)  # [batch_size, 2, embed_dim]

        # Reformat for multi-head attention (seq_len, batch_size, embed_dim)
        combined = combined.permute(1, 0, 2)

        # Apply self-attention
        attn_output, _ = self.self_attn(combined, combined, combined)
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, 2, embed_dim]
        attn_output = self.attn_drop(attn_output)

        # Split attended embeddings
        e1_attn = attn_output[:, 0, :]
        rel_attn = attn_output[:, 1, :]

        # ========== Convolution Branch ==========
        # Reshape attended embeddings for convolution
        e1_reshaped = e1_attn.view(-1, 1, self.embedding_shape1, self.embedding_shape2)
        rel_reshaped = rel_attn.view(-1, 1, self.embedding_shape1, self.embedding_shape2)

        # Stack along height dimension
        stacked_inputs = torch.cat([e1_reshaped, rel_reshaped], 2)

        # Batch normalization and dropout
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)

        # Convolution operations
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        # Prepare for final scoring
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Score against all entities
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b

        return x