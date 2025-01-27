# =============================================================================
# Created/Modified files during execution:
print("calp.py")
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnConvE(nn.Module):


    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        num_attention_heads: int = 4,
        ff_hidden_dim: int = 400,  # hidden dim in FFN after attention
        conv_channels: int = 32,
        embedding_shape1: int = 20,
        use_stacked_embeddings: bool = True,
        dropout_attention: float = 0.1,   # dropout for attention and FFN
        dropout_input: float = 0.2,      # dropout before convolution
        dropout_feature_map: float = 0.2 # dropout after convolution
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        # For the 2D reshape
        self.embedding_shape1 = embedding_shape1
        self.embedding_shape2 = embedding_dim // embedding_shape1
        self.use_stacked_embeddings = use_stacked_embeddings

        # Embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        #
        # -------------------- SELF-ATTENTION BRANCH --------------------
        #
        # 1) Multi-Head Self Attention
        #
        # PyTorch's nn.MultiheadAttention expects shape (L, N, E) => (seq_len=2, batch_size, embed_dim)
        #
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout_attention,         # applies dropout to attention output
            batch_first=False                  # we will pass (2, batch_size, d)
        )
        # 2) Post-attention feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_attention),
            nn.Linear(ff_hidden_dim, embedding_dim),
            nn.Dropout(dropout_attention)
        )
        # Layer norm for post-attention and post-ffn
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        #
        # -------------------- 2D CONVOLUTION BRANCH --------------------
        #
        # We reshape e_s & e_r into two 2D tensors => concat => conv2d => flatten => linear => fconv
        #
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=(3,3), padding=1)
        self.bn_conv = nn.BatchNorm2d(conv_channels)
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout_feature_map = nn.Dropout(dropout_feature_map)

        # The height of the 2D input
        if use_stacked_embeddings:
            conv_output_height = 2 * self.embedding_shape1  # stacked
        else:
            conv_output_height = 2 * self.embedding_shape1  # if you do interleaving, same dimension

        conv_output_width = self.embedding_shape2
        fc_length = conv_channels * conv_output_height * conv_output_width
        self.fc = nn.Linear(fc_length, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

        #
        # -------------------- FUSION LAYER --------------------
        #
        # We have Fs = e_s' + α * Fconv
        #         Fr = e_r' + β * Fconv
        #
        # α, β are learnable scalars
        #
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

    def _apply_self_attention(self, e_s, e_r):
        """
        e_s, e_r: (batch_size, embedding_dim)
        Returns updated e_s', e_r' using multi-head self-attention and feed-forward.
        """
        batch_size = e_s.size(0)
        d = e_s.size(1)

        # Stack up as a sequence of length 2: shape => (2, batch_size, d)
        # (We place subject as index 0, relation as index 1)
        seq = torch.stack([e_s, e_r], dim=0)  # [2, B, d]

        # Multi-head attention:
        # We'll treat the entire [2, B, d] as both "query", "key", and "value"
        attn_output, _ = self.self_attention(seq, seq, seq)  # shape => (2, B, d)
        # Residual + norm
        seq = self.norm1(seq + attn_output)

        # Feed forward
        ff_output = self.ffn(seq)  # shape => (2, B, d)
        seq = self.norm2(seq + ff_output)

        # Extract e_s' and e_r' from the final (2, B, d)
        e_s_out = seq[0, :, :]  # (B, d)
        e_r_out = seq[1, :, :]  # (B, d)
        return e_s_out, e_r_out

    def _apply_2d_conv(self, e_s, e_r):
        """
        e_s, e_r: (batch_size, embedding_dim)
        Returns fconv: (batch_size, embedding_dim)
        capturing local feature interactions through 2D conv.
        """
        batch_size = e_s.size(0)

        # Reshape for 2D convolution
        # e_s => (batch_size, 1, embedding_shape1, embedding_shape2)
        # e_r => likewise
        e_s_2d = e_s.view(-1, self.embedding_shape1, self.embedding_shape2)
        e_r_2d = e_r.view(-1, self.embedding_shape1, self.embedding_shape2)

        if self.use_stacked_embeddings:
            # Stack along height => shape = [batch_size, 1, 2*embedding_shape1, embedding_shape2]
            stacked_inputs = torch.cat([e_s_2d, e_r_2d], dim=1).unsqueeze(1)
        else:
            # Interleave approach
            interleaved = torch.zeros_like(e_s_2d.repeat(1, 2, 1))
            interleaved[:, 0::2, :] = e_s_2d
            interleaved[:, 1::2, :] = e_r_2d
            stacked_inputs = interleaved.unsqueeze(1)  # => (batch_size, 1, 2*embed_shape1, embed_shape2)

        # Dropout before convolution
        x = self.dropout_input(stacked_inputs)

        # 2D convolution
        x = self.conv2d(x)                   # => (batch_size, conv_channels, 2*emb_shape1, emb_shape2)
        x = self.bn_conv(x)
        x = F.relu(x)
        x = self.dropout_feature_map(x)

        # Flatten
        batch_size, c, h, w = x.shape
        x = x.view(batch_size, c*h*w)

        # FC layer
        x = self.fc(x)                        # => (batch_size, embedding_dim)
        x = self.bn_fc(x)
        x = F.relu(x)

        return x

    def forward(self, subject_idx, relation_idx):
        """
        Forward pass to predict the tail entity.
        Returns: (batch_size, num_entities) scores
        """
        batch_size = subject_idx.size(0)

        # Get the subject, relation embeddings
        e_s = self.entity_embeddings(subject_idx)    # (B, d)
        e_r = self.relation_embeddings(relation_idx) # (B, d)

        # ----------------- Self-Attention Branch -----------------
        e_s_att, e_r_att = self._apply_self_attention(e_s, e_r)

        # ----------------- 2D Convolution Branch -----------------
        fconv = self._apply_2d_conv(e_s, e_r)

        # ----------------- Fusion -----------------
        # Fs = e_s_att + alpha*fconv
        # Fr = e_r_att + beta*fconv
        Fs = e_s_att + self.alpha * fconv
        Fr = e_r_att + self.beta  * fconv

        # For "TwoMult" => elementwise multiplication of Fs and Fr
        # Then compare with all entity embeddings
        # phi(s,r,t) = (Fr)ᵀ e_t in the paper, but one can also do Fs ⊙ Fr
        # Here we'll do a small variation: Fs ⊙ Fr => shape (B, d)
        x = Fs * Fr

        # Scores: (B, num_entities)
        # matrix multiply with all entity embeddings
        scores = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))

        return scores

    def forward_head(self, tail_idx, relation_idx):
        """
        Forward pass to predict the head entity.
        Similar to forward() but now we input the tail index and the relation index.
        """
        batch_size = tail_idx.size(0)

        e_t = self.entity_embeddings(tail_idx)      # (B, d)
        e_r = self.relation_embeddings(relation_idx)# (B, d)

        # Self-attention branch
        e_t_att, e_r_att = self._apply_self_attention(e_t, e_r)

        # 2D convolution branch
        fconv = self._apply_2d_conv(e_t, e_r)

        # Fusion
        Ft = e_t_att + self.alpha * fconv
        Fr = e_r_att + self.beta  * fconv

        # Combine
        x = Ft * Fr

        scores = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        return scores