import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvE(nn.Module):
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
        use_bias: bool = True,
        num_filters: int = 32
    ):
        super().__init__()

        # Embeddings
        self.emb_e = nn.Embedding(num_entities, embedding_dim, padding_idx=0)
        self.emb_rel = nn.Embedding(num_relations, embedding_dim, padding_idx=0)

        # Dimensions for reshaping
        self.embedding_dim = embedding_dim
        self.embedding_shape1 = embedding_shape1
        self.embedding_shape2 = embedding_dim // embedding_shape1

        # Dropouts
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)

        # First convolution layer
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=(3, 3), stride=1, padding=0, bias=use_bias)
        # Second convolution layer
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=(3, 3), stride=1, padding=1, bias=use_bias)

        # Batch norm layers
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.bn3 = nn.BatchNorm1d(embedding_dim)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, embedding_dim)

        # Bias for entities
        self.b = nn.Parameter(torch.zeros(num_entities))

        self.init()

    def init(self):
        """Initialize embeddings."""
        nn.init.xavier_normal_(self.emb_e.weight.data)
        nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        # Reshape entity and relation embeddings to 2D
        e1_emb = self.emb_e(e1).view(-1, 1, self.embedding_shape1, self.embedding_shape2)
        rel_emb = self.emb_rel(rel).view(-1, 1, self.embedding_shape1, self.embedding_shape2)

        # Concatenate along "height" dimension
        stacked_inputs = torch.cat([e1_emb, rel_emb], dim=2)  # shape: [batch, 1, 2*shape1, shape2]

        # Batch norm + input dropout
        x = self.bn0(stacked_inputs)
        x = self.inp_drop(x)

        # First conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)  # dropout on feature maps

        # Second conv layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        # Flatten
        x = x.view(x.shape[0], -1)

        # Fully connected layer
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Match against all entities
        scores = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        scores += self.b.expand_as(scores)  # add bias

        # Sigmoid for final prediction
        pred = torch.sigmoid(scores)
        return pred

    def score_triple(self, e1, rel, e2):
        """Convenience method to get the score for a single triple."""
        scores = self.forward(e1, rel)
        return scores.gather(1, e2.view(-1, 1))
