import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEDeep(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        embedding_shape1: int = 20,
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

        # Dropout layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.feature_map_dropout = nn.Dropout(feature_map_dropout)

        # ---------------------------------------------------------------------
        # Convolution layers (deeper model)
        # ---------------------------------------------------------------------
        # First convolution layer
        self.conv2d_1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1
        )
        self.bn_c1 = nn.BatchNorm2d(32)

        # Second convolution layer
        self.conv2d_2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1
        )
        self.bn_c2 = nn.BatchNorm2d(64)

        # ---------------------------------------------------------------------
        # We keep embedding stacking/interleaving logic the same
        # ---------------------------------------------------------------------
        if use_stacked_embeddings:
            conv_output_height = 2 * self.embedding_shape1
        else:
            conv_output_height = 2 * self.embedding_shape1

        conv_output_width = self.embedding_shape2

        # ---------------------------------------------------------------------
        # Fully-connected layer
        # After the second Conv2D, we have 64 feature maps
        # The flattened size is (64 * conv_output_height * conv_output_width)
        # ---------------------------------------------------------------------
        fc_length = 64 * conv_output_height * conv_output_width
        self.fc = nn.Linear(fc_length, embedding_dim)

        # BatchNorm for the FC output
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

        # Initialize embeddings
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    def _reshape_embeddings(self, e1_embedded, rel_embedded):
        """
        Reshapes and either stacks or interleaves the subject and relation
        embeddings for convolution input.
        """
        if self.use_stacked_embeddings:
            # Original stacked version
            stacked_inputs = torch.stack([e1_embedded, rel_embedded], dim=1)
            stacked_inputs = stacked_inputs.reshape(
                -1, 1, 2 * self.embedding_shape1, self.embedding_shape2
            )
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
        Forward pass to predict the scores for all possible objects
        given (subject, relation).
        """
        batch_size = subject_idx.size(0)

        # Get embeddings
        subject_embedded = self.entity_embeddings(subject_idx).view(-1, self.embedding_dim)
        relation_embedded = self.relation_embeddings(relation_idx).view(-1, self.embedding_dim)

        # Reshape
        stacked_inputs = self._reshape_embeddings(subject_embedded, relation_embedded)

        # Apply input dropout
        x = self.input_dropout(stacked_inputs)

        # ---------------------------------------------------------------------
        # Pass through first conv layer
        # ---------------------------------------------------------------------
        x = self.conv2d_1(x)
        x = self.bn_c1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        # ---------------------------------------------------------------------
        # Pass through second conv layer
        # ---------------------------------------------------------------------
        x = self.conv2d_2(x)
        x = self.bn_c2(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        # Flatten
        x = x.view(batch_size, -1)

        # FC layer
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn_fc(x)
        x = F.relu(x)

        # Score computation against all entities
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        return x

    def forward_head(self, tail_idx, relation_idx):
        """
        Forward pass for head prediction tasks.
        Predict scores for all possible subjects given (tail, relation).
        """
        batch_size = tail_idx.size(0)

        # Get embeddings
        tail_embedded = self.entity_embeddings(tail_idx).view(-1, self.embedding_dim)
        relation_embedded = self.relation_embeddings(relation_idx).view(-1, self.embedding_dim)

        # Reshape
        stacked_inputs = self._reshape_embeddings(tail_embedded, relation_embedded)

        # Apply input dropout
        x = self.input_dropout(stacked_inputs)

        # First conv layer
        x = self.conv2d_1(x)
        x = self.bn_c1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        # Second conv layer
        x = self.conv2d_2(x)
        x = self.bn_c2(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        # Flatten
        x = x.view(batch_size, -1)

        # FC layer
        x = self.fc(x)
        x = self.hidden_dropout(x)
        x = self.bn_fc(x)
        x = F.relu(x)

        # Score computation
        x = torch.mm(x, self.entity_embeddings.weight.transpose(1, 0))
        return x