import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter


class DeepConvE(nn.Module):
    """
    Deep Convolutional Neural Network model for Knowledge Graph Link Prediction.
    An enhanced version of ConvE with deeper architecture and residual connections.

    Args:
        config (dict): Configuration dictionary containing model hyperparameters
        num_entities (int): Total number of entities in the knowledge graph
        num_relations (int): Total number of relations in the knowledge graph
    """
    def __init__(self, config, num_entities, num_relations):
        super().__init__()

        self.emb_e = nn.Embedding(num_entities, config['embedding_dim'], padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, config['embedding_dim'], padding_idx=0)

        self.inp_drop = nn.Dropout(config['input_drop'])
        self.hidden_drop = nn.Dropout(config['hidden_drop'])
        self.feature_map_drop = nn.Dropout2d(config['feat_drop'])

        self.emb_dim1 = config['embedding_shape1']
        self.emb_dim2 = config['embedding_dim'] // self.emb_dim1

        self.use_stacked_embeddings = True
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 32, (3, 3), 1, 1, bias=config['use_bias']),
            nn.Conv2d(32, 64, (3, 3), 1, 1, bias=config['use_bias']),
            nn.Conv2d(64, 128, (3, 3), 1, 1, bias=config['use_bias']),
            nn.Conv2d(128, 256, (3, 3), 1, 1, bias=config['use_bias'])
        ])

        self.bn0 = nn.BatchNorm2d(1)
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256)
        ])
        self.bn_final = nn.BatchNorm1d(config['embedding_dim'])

        conv_output_size = self._calculate_conv_output_size()
        self.fc_layers = nn.ModuleList([
            nn.Linear(conv_output_size, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, config['embedding_dim'])
        ])

        self.b = Parameter(torch.zeros(num_entities))

        self.init()

    def _calculate_conv_output_size(self):
        """
        Calculates the output size after the convolution layers.
        
        Returns:
            int: Size of the flattened feature map after convolutions
        """
        h = self.emb_dim1 * 2
        w = self.emb_dim2
        return 256 * h * w

    def init(self):
        """
        Initializes model weights using Xavier normal initialization.
        """
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        for conv in self.conv_layers:
            xavier_normal_(conv.weight.data)
        for fc in self.fc_layers:
            xavier_normal_(fc.weight.data)

    def arrange_embeddings(self, e1_embedded, rel_embedded):
        """
        Arranges entity and relation embeddings in the required format.

        Args:
            e1_embedded: Entity embeddings
            rel_embedded: Relation embeddings

        Returns:
            torch.Tensor: Arranged embeddings ready for convolution
        """
        if self.use_stacked_embeddings:
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
            e1: Entity indices
            rel: Relation indices

        Returns:
            torch.Tensor: Predicted scores for all possible tail entities
        """
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)

        stacked_inputs = self.arrange_embeddings(e1_embedded, rel_embedded)
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.feature_map_drop(x)

        x = x.view(x.shape[0], -1)

        for i, fc in enumerate(self.fc_layers):
            identity = x
            x = fc(x)
            x = F.relu(x)
            x = self.hidden_drop(x)
            if i < len(self.fc_layers) - 1 and identity.shape == x.shape:
                x = x + identity

        x = self.bn_final(x)
        x = F.relu(x)

        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x = x + self.b

        return torch.sigmoid(x)

    def loss(self, pred, target):
        """
        Computes the binary cross entropy loss.

        Args:
            pred: Predicted scores
            target: Target values

        Returns:
            torch.Tensor: Computed loss value
        """
        return F.binary_cross_entropy(pred, target)