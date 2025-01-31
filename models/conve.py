import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter


class ConvE(nn.Module):
    def __init__(self, config, num_entities, num_relations):
        super().__init__()

        # Embeddings
        self.emb_e = nn.Embedding(num_entities, config['embedding_dim'], padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, config['embedding_dim'], padding_idx=0)

        # Dropout layers
        self.inp_drop = nn.Dropout(config['input_drop'])
        self.hidden_drop = nn.Dropout(config['hidden_drop'])
        self.feature_map_drop = nn.Dropout2d(config['feat_drop'])

        # Dimensions
        self.emb_dim1 = config['embedding_shape1']
        self.emb_dim2 = config['embedding_dim'] // self.emb_dim1

        # Embedding arrangement
        if config['embedding_style'] == 'stacked':
            self.alternate_embeddings = False
        elif config['embedding_style'] == 'alternating':
            self.alternate_embeddings = True
        else:
            raise ValueError(f"Invalid embedding style: {config['embedding_style']}")

        # Layers
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=config['use_bias'])
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(config['embedding_dim'])
        self.fc = nn.Linear(config['hidden_size'], config['embedding_dim'])

        # Bias
        self.b = Parameter(torch.zeros(num_entities))

        # Initialize weights
        self.init()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def arrange_embeddings(self, e1_embedded, rel_embedded):
        if not self.alternate_embeddings:
            # Original stacked version
            return torch.cat([e1_embedded, rel_embedded], 2)
        else:
            # Alternating version
            batch_size = e1_embedded.size(0)
            # Reshape to (batch_size, rows, emb_dim2)
            e1_rows = e1_embedded.view(batch_size, self.emb_dim1, self.emb_dim2)
            rel_rows = rel_embedded.view(batch_size, self.emb_dim1, self.emb_dim2)

            # Create empty tensor for interleaved result
            total_rows = self.emb_dim1 * 2
            result = torch.empty(batch_size, total_rows, self.emb_dim2,
                               device=e1_embedded.device)

            # Fill alternating rows
            result[:, 0::2] = e1_rows
            result[:, 1::2] = rel_rows

            # Reshape back to (batch_size, 1, total_rows, emb_dim2)
            return result.unsqueeze(1)

    def forward(self, e1, rel):
        # Get embeddings and reshape
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        # Arrange embeddings (stacked or alternating)
        stacked_inputs = self.arrange_embeddings(e1_embedded, rel_embedded)
        stacked_inputs = self.bn0(stacked_inputs)

        # Apply convolution
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        # Reshape and apply linear
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Final computations
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x = x + self.b

        return torch.sigmoid(x)

    def loss(self, pred, target):
        return F.binary_cross_entropy(pred, target)