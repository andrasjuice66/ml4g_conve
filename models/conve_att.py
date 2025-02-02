import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter


class AttnConvE(nn.Module):
    def __init__(self, config, num_entities, num_relations):
        super().__init__()

        self.embedding_dim = config['embedding_dim']
        self.emb_dim1 = config['embedding_shape1']
        self.emb_dim2 = config['embedding_dim'] // self.emb_dim1
        self.use_stacked_embeddings = True

        self.num_attention_heads = config['num_attention_heads']
        self.ff_hidden_dim = config['ff_hidden_dim']

        self.dropout_attention = config['dropout_attention']
        self.dropout_input = config['dropout_input']
        self.dropout_feature = config['dropout_feature']

        self.emb_e = nn.Embedding(num_entities, self.embedding_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, self.embedding_dim, padding_idx=0)

        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_attention_heads,
            dropout=self.dropout_attention,
            batch_first=False
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.embedding_dim, self.ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_attention),
            nn.Linear(self.ff_hidden_dim, self.embedding_dim),
            nn.Dropout(self.dropout_attention)
        )

        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)

        self.inp_drop = nn.Dropout(self.dropout_input)
        self.hidden_drop = nn.Dropout(self.dropout_attention)
        self.feature_map_drop = nn.Dropout2d(self.dropout_feature)

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, padding=1, bias=config['use_bias'])
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)

        conv_output_height = 2 * self.emb_dim1
        conv_output_width = self.emb_dim2
        fc_length = 32 * conv_output_height * conv_output_width
        self.fc = nn.Linear(fc_length, self.embedding_dim)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.b = Parameter(torch.zeros(num_entities))

        self.init()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def _apply_attention(self, e_s, e_r):
        seq = torch.stack([e_s, e_r], dim=0)
        attn_output, _ = self.self_attention(seq, seq, seq)
        seq = self.norm1(seq + attn_output)

        ff_output = self.ffn(seq)
        seq = self.norm2(seq + ff_output)

        return seq[0], seq[1]

    def _arrange_embeddings(self, e_s, e_r):
        if self.use_stacked_embeddings:
            return torch.cat([e_s, e_r], dim=2)
        else:
            batch_size = e_s.size(0)
            e_s_2d = e_s.view(batch_size, self.emb_dim1, self.emb_dim2)
            e_r_2d = e_r.view(batch_size, self.emb_dim1, self.emb_dim2)

            result = torch.empty(batch_size, 2 * self.emb_dim1, self.emb_dim2,
                               device=e_s.device)
            result[:, 0::2] = e_s_2d
            result[:, 1::2] = e_r_2d
            return result.unsqueeze(1)

    def _apply_convolution(self, e_s, e_r):
        stacked_inputs = self._arrange_embeddings(e_s, e_r)
        stacked_inputs = self.bn0(stacked_inputs)

        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

    def forward(self, e1, rel):
        e_s = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        e_r = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        e_s_att, e_r_att = self._apply_attention(
            e_s.view(e_s.size(0), -1),
            e_r.view(e_r.size(0), -1)
        )

        f_conv = self._apply_convolution(e_s, e_r)

        f_s = e_s_att + self.alpha * f_conv
        f_r = e_r_att + self.beta * f_conv

        x = f_s * f_r
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x = x + self.b

        return torch.sigmoid(x)

    def loss(self, pred, target):
        return F.binary_cross_entropy(pred, target)