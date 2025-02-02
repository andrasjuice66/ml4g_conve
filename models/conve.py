import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter


class ConvE(nn.Module):
    """
    Convolutional Neural Network model for Knowledge Graph Link Prediction.
    Implementation of the ConvE model as described in the paper
    "Convolutional 2D Knowledge Graph Embeddings".

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        num_entities (int): Total number of entities in the knowledge graph
        num_relations (int): Total number of relations in the knowledge graph
        embedding_style (str): Style of embedding arrangement ('stacked' or 'alternating')
    """
    def __init__(self, config, num_entities, num_relations, embedding_style):
        super().__init__()

        self.emb_e = nn.Embedding(num_entities, config['embedding_dim'], padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, config['embedding_dim'], padding_idx=0)

        self.inp_drop = nn.Dropout(config['input_drop'])
        self.hidden_drop = nn.Dropout(config['hidden_drop'])
        self.feature_map_drop = nn.Dropout2d(config['feat_drop'])

        self.emb_dim1 = config['embedding_shape1']
        self.emb_dim2 = config['embedding_dim'] // self.emb_dim1

        if embedding_style == 'stacked':
            self.alternate_embeddings = False
        elif embedding_style == 'alternating':
            self.alternate_embeddings = True
        else:
            raise ValueError(f"Invalid embedding style: {config['embedding_style']}")

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=config['use_bias'])
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(config['embedding_dim'])
        self.fc = nn.Linear(config['hidden_size'], config['embedding_dim'])

        self.b = Parameter(torch.zeros(num_entities))

        self.init()

    def init(self):
        """
        Initializes model weights using Xavier normal initialization.
        """
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def arrange_embeddings(self, e1_embedded, rel_embedded):
        """
        Arranges entity and relation embeddings in either stacked or alternating format.

        Args:
            e1_embedded (torch.Tensor): Entity embeddings
            rel_embedded (torch.Tensor): Relation embeddings

        Returns:
            torch.Tensor: Arranged embeddings ready for convolution
        """
        if not self.alternate_embeddings:
            return torch.cat([e1_embedded, rel_embedded], 2)
        else:
            batch_size = e1_embedded.size(0)
            e1_rows = e1_embedded.view(batch_size, self.emb_dim1, self.emb_dim2)
            rel_rows = rel_embedded.view(batch_size, self.emb_dim1, self.emb_dim2)

            total_rows = self.emb_dim1 * 2
            result = torch.empty(batch_size, total_rows, self.emb_dim2,
                               device=e1_embedded.device)

            result[:, 0::2] = e1_rows
            result[:, 1::2] = rel_rows

            return result.unsqueeze(1)

    def forward(self, e1, rel):
        """
        Forward pass of the model.

        Args:
            e1 (torch.Tensor): Entity indices
            rel (torch.Tensor): Relation indices

        Returns:
            torch.Tensor: Predicted scores for all possible tail entities
        """
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = self.arrange_embeddings(e1_embedded, rel_embedded)
        stacked_inputs = self.bn0(stacked_inputs)

        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x = x + self.b

        return torch.sigmoid(x)

    def loss(self, pred, target):
        """
        Computes the binary cross entropy loss.

        Args:
            pred (torch.Tensor): Predicted scores
            target (torch.Tensor): Target values

        Returns:
            torch.Tensor: Computed loss value
        """
        return F.binary_cross_entropy(pred, target)