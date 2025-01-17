import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    """
    A single-head attention module for combining e1 and r embeddings.
    For multi-head, you can stack multiple heads and concatenate.
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Linear transformations for query, key, and value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, e1_emb, rel_emb):
        """
        e1_emb, rel_emb: (batch, embed_dim)
        We'll treat e1_emb as queries and rel_emb as keys/values, or vice versa.
        """
        # Convert to queries, keys, values
        Q = self.query(e1_emb)   # (batch, embed_dim)
        K = self.key(rel_emb)    # (batch, embed_dim)
        V = self.value(rel_emb)  # (batch, embed_dim)

        # Compute attention scores: Q*K^T
        # But first, let's reshape to (batch, 1, embed_dim) for matrix multiplication
        Q_ = Q.unsqueeze(1)  # (batch, 1, embed_dim)
        K_ = K.unsqueeze(2)  # (batch, embed_dim, 1)

        # attention_scores shape: (batch, 1, 1)
        attention_scores = torch.bmm(Q_, K_)

        # scale by sqrt(embed_dim) if needed
        scale = Q.size(-1) ** 0.5
        attention_scores = attention_scores / scale

        # Softmax over the "key" dimension (which is 1 here, trivial in this simplistic approach)
        # In a multi-token scenario, you'd have more keys to attend to.
        attn_weights = self.softmax(attention_scores)  # (batch, 1, 1)

        # Weighted sum of V: again, shape will be (batch, 1, embed_dim) after bmm
        V_ = V.unsqueeze(1)  # (batch, 1, embed_dim)
        attended = torch.bmm(attn_weights, V_)  # (batch, 1, embed_dim)

        # Squeeze back to (batch, embed_dim)
        attended = attended.squeeze(1)
        return attended
    

class AttnConvE(nn.Module):
    def __init__(
        self, 
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,
        hidden_size: int = 9728,
        input_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        feature_map_dropout: float = 0.2,
        use_bias: bool = True
    ):
        super().__init__()
        
        # Embeddings
        self.emb_e = nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, embedding_dim, padding_idx=0)
        
        self.embedding_dim = embedding_dim
        self.embedding_shape1 = embedding_shape1
        self.embedding_shape2 = embedding_dim // embedding_shape1
        
        # Attention module
        self.attention = SimpleAttention(embedding_dim)

        # Dropouts
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)
        
        # Convolution
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=use_bias)
        
        # Batch norms
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        
        # Fully connected
        self.fc = nn.Linear(hidden_size, embedding_dim)
        
        # Entity bias
        self.b = nn.Parameter(torch.zeros(num_entities))
        
        self.init()

    def init(self):
        """Initialize embeddings."""
        nn.init.xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        # Lookup embeddings: (batch, embed_dim)
        e1_emb = self.emb_e(e1)
        rel_emb = self.emb_rel(rel)
        
        # === 1. Attention ===
        # Combine e1_emb and rel_emb into a single attended vector
        # For instance, we treat e1 as query, rel as key/value
        attended_rel = self.attention(e1_emb, rel_emb)  # (batch, embed_dim)
        
        # You can choose to combine e1 and attended_rel in various ways:
        # Option A: Just replace the relation with the attended version
        # Option B: Concatenate e1 and attended_rel, then reshape
        # We'll do Option A here for simplicity.
        
        # Reshape the embeddings for convolution
        # entity => (batch, 1, shape1, shape2)
        e1_emb_2d = e1_emb.view(-1, 1, self.embedding_shape1, self.embedding_shape2)
        # attended relation => (batch, 1, shape1, shape2)
        rel_emb_2d = attended_rel.view(-1, 1, self.embedding_shape1, self.embedding_shape2)

        # Concatenate along the height dimension
        stacked_inputs = torch.cat([e1_emb_2d, rel_emb_2d], dim=2)  # shape: [batch, 1, shape1*2, shape2]
        
        # === 2. Continue with standard ConvE pipeline ===
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Compute scores
        scores = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        scores += self.b.expand_as(scores)
        
        preds = torch.sigmoid(scores)
        return preds

    def score_triple(self, e1, rel, e2):
        """Convenience method to get the score for a single triple."""
        scores = self.forward(e1, rel)
        return scores.gather(1, e2.view(-1, 1))
